##############################
# INSTALL LIBRARIES
##############################
!pip install git+https://github.com/huggingface/transformers.git@main
!pip install -q datasets
!pip install rouge-score
!pip install --upgrade datasets
!pip install huggingface_hub datasets
!pip install "kagglehub[pandas-datasets]"
!pip install bitsandbytes
!pip install peft

!huggingface-cli login --token TOKEN_HERE

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter


import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoModelForVision2Seq, AutoProcessor
from datasets import load_dataset
import kagglehub
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import os
from huggingface_hub import login

# Log in to Hugging Face (optional, for pushing to Hub)
login(token="TOKEN_HERE")

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor with proper initialization
print("Loading SmolVLM model and processor...")
model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM-500M-Instruct",
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(
    "HuggingFaceTB/SmolVLM-500M-Instruct",
    size={"longest_edge": 512},
    trust_remote_code=True
)

# Load dataset and captions
dataset = load_dataset("eltorio/ROCOv2-radiology")

# Assign each split to a variable
train_dataset = dataset["train"]
val_dataset = dataset["validation"]
test_dataset = dataset["test"]

# Print out the number of examples in each
print(f"Train set size:      {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size:       {len(test_dataset)}")

# Define custom dataset class
class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        caption = item["caption"]
        return image, caption

# Updated collate function
def smolvlm_collate_fn(batch):
    images = [item[0] for item in batch]
    captions = [item[1] for item in batch]

    # Create messages for each sample
    messages_list = []
    for image, caption in zip(images, captions):
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Describe this image:"}]},
            {"role": "assistant", "content": [{"type": "text", "text": caption}]}
        ]
        messages_list.append(messages)

    # Apply chat template
    texts = [processor.apply_chat_template(messages, tokenize=False) for messages in messages_list]

    # Process texts and images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # Set labels
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == image_token_id] = -100
    batch["labels"] = labels

    # Move to device
    batch = {k: v.to(device) for k, v in batch.items()}

    return batch

# Create dataloaders
train_dataset = ImageCaptioningDataset(train_dataset)
val_dataset = ImageCaptioningDataset(val_dataset)
test_dataset = ImageCaptioningDataset(test_dataset)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=smolvlm_collate_fn)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=2, collate_fn=smolvlm_collate_fn)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=2, collate_fn=smolvlm_collate_fn)

# Define optimizer and scheduler
optimizer = Adam(model.parameters(), lr=5e-5)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

# Create checkpoint directory
checkpoint_dir = "/kaggle/working/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Validation loss computation
def compute_val_loss(model, val_dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(val_dataloader)

# Test loss computation
def compute_test_loss(model, test_dataloader, device):
    return compute_val_loss(model, test_dataloader, device)

# Training loop with checkpointing
num_epochs = 4
for epoch in range(num_epochs):
    model.train()
    for idx, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        print(f"Epoch {epoch}, Batch {idx}, Training Loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Compute validation loss
    val_loss = compute_val_loss(model, val_dataloader, device)
    print(f"Epoch {epoch}, Validation Loss: {val_loss}")
    scheduler.step(val_loss)

    # Save model and processor checkpoint
    epoch_dir = os.path.join(checkpoint_dir, f"epoch_{epoch+1}")
    os.makedirs(epoch_dir, exist_ok=True)
    model.save_pretrained(epoch_dir)
    processor.save_pretrained(epoch_dir)
    print(f"Checkpoint saved for epoch {epoch+1} at {epoch_dir}")

    # Optionally push to Hugging Face Hub
    hub_repo = f"nafew/smolvlm-500m-instruct-roco-epoch-{epoch+1}"
    model.push_to_hub(hub_repo)
    processor.push_to_hub(hub_repo)
    print(f"Checkpoint for epoch {epoch+1} pushed to Hugging Face: {hub_repo}")

# Compute test loss
test_loss = compute_test_loss(model, test_dataloader, device)
print(f"Test Loss: {test_loss}")

# Save final model and processor
final_dir = "/kaggle/working/finetuned_smolvlm_500m_instruct-roco-final_10_epoch"
os.makedirs(final_dir, exist_ok=True)
model.save_pretrained(final_dir)
processor.save_pretrained(final_dir)
print(f"Final model and processor saved to {final_dir}")

# Push final model to Hugging Face Hub
final_hub_repo = "nafew/smolvlm-500m-instruct-roco-final_10_epoch"
model.push_to_hub(final_hub_repo)
processor.push_to_hub(final_hub_repo)
print(f"Final model and processor pushed to Hugging Face: {final_hub_repo}")
