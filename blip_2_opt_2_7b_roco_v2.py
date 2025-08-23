##############################
# INSTALL LIBRARIES
##############################
!pip install git+https://github.com/huggingface/transformers.git@main
!pip install -q datasets
!pip install rouge-score
!pip install --upgrade datasets
!rm -rf ~/.cache/huggingface/datasets/alesanm___balenciaga_short_descriptions
!pip install huggingface_hub datasets
!huggingface-cli login --token TOKEN_HERE
!pip install peft
!pip install bitsandbytes

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
import nltk
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score
import torch.nn.functional as F
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoImageProcessor, ViTModel
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoProcessor, Blip2ForConditionalGeneration, AutoTokenizer
import time
import random
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import BitsAndBytesConfig
import warnings
from nltk.translate.meteor_score import meteor_score
import os

warnings.filterwarnings("ignore")

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = load_dataset("eltorio/ROCOv2-radiology")
train_dataset = dataset["train"]
val_dataset = dataset["validation"]
test_dataset = dataset["test"]

# Define custom dataset class
class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(
            images=item["image"],
            text=item["caption"],
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        return encoding

# Initialize processor and base model with quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

processor_tuned = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
base_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    quantization_config=quantization_config,
    device_map={"": 0},
)

# Set up LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
)

# Resuming logic
resume_from = None  # Set to the path of the checkpoint directory to resume from, e.g., "/kaggle/working/checkpoints/epoch_5"

if resume_from is not None:
    # Load the adapter
    model_tuned = PeftModel.from_pretrained(base_model, resume_from)
    # Load training state
    checkpoint = torch.load(os.path.join(resume_from, 'training_state.pt'))
    start_epoch = checkpoint['epoch'] + 1
    optimizer = AdamW(model_tuned.parameters(), lr=2e-5)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
else:
    # Apply LoRA
    model_tuned = get_peft_model(base_model, lora_config)
    start_epoch = 0
    optimizer = AdamW(model_tuned.parameters(), lr=2e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

# Create datasets and dataloaders
train_dataset = ImageCaptioningDataset(train_dataset, processor_tuned)
val_dataset = ImageCaptioningDataset(val_dataset, processor_tuned)
test_dataset = ImageCaptioningDataset(test_dataset, processor_tuned)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=4)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=4)

# Create checkpoint directory
checkpoint_dir = "/kaggle/working/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Validation and test loss functions
def compute_val_loss(model, val_dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(val_dataloader)

def compute_test_loss(model, test_dataloader, device):
    return compute_val_loss(model, test_dataloader, device)

# Training loop
num_epochs = 10
for epoch in range(start_epoch, num_epochs):
    model_tuned.train()
    for idx, batch in enumerate(train_dataloader):
        input_ids = batch["input_ids"].to("cuda:0")
        pixel_values = batch["pixel_values"].to("cuda:0")
        outputs = model_tuned(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)
        loss = outputs.loss
        print(f"Epoch {epoch}, Batch {idx}, Training Loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    val_loss = compute_val_loss(model_tuned, val_dataloader, "cuda:0")
    print(f"Epoch {epoch}, Validation Loss: {val_loss}")
    scheduler.step(val_loss)

    # Save checkpoint locally
    epoch_dir = os.path.join(checkpoint_dir, f"epoch_{epoch+1}")
    os.makedirs(epoch_dir, exist_ok=True)
    model_tuned.save_pretrained(epoch_dir)
    torch.save({
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
    }, os.path.join(epoch_dir, 'training_state.pt'))
    print(f"LoRA checkpoint saved locally for epoch {epoch+1} at {epoch_dir}")

    # Push LoRA-adapted model and processor to Hugging Face
    lora_repo_id = f"blip2-opt-2.7b-roco-lora-epoch-{epoch+1}"
    model_tuned.push_to_hub(lora_repo_id)
    processor_tuned.push_to_hub(lora_repo_id)
    print(f"LoRA checkpoint for epoch {epoch+1} pushed to Hugging Face: {lora_repo_id}")

    # Optionally, save and push merged model for inference
    merged_model = model_tuned.merge_and_unload()
    merged_dir = os.path.join(epoch_dir, 'merged')
    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(merged_dir)
    processor_tuned.save_pretrained(merged_dir)
    merged_repo_id = f"blip2-opt-2.7b-roco-merged-epoch-{epoch+1}"
    merged_model.push_to_hub(merged_repo_id)
    processor_tuned.push_to_hub(merged_repo_id)
    print(f"Merged checkpoint for epoch {epoch+1} pushed to Hugging Face: {merged_repo_id}")

# Evaluate on test set
test_loss = compute_test_loss(model_tuned, test_dataloader, "cuda:0")
print(f"Test Loss: {test_loss}")

# Save final merged model and processor
final_dir = "/kaggle/working/finetuned_blip2_opt_2.7b_10_epoch"
os.makedirs(final_dir, exist_ok=True)
merged_model = model_tuned.merge_and_unload()
merged_model.save_pretrained(final_dir)
processor_tuned.save_pretrained(final_dir)
print(f"Final merged model and processor saved to {final_dir}")

# Push final merged model to Hugging Face
final_hub_repo = "blip2-opt-2.7b-roco-final_10_epoch"
merged_model.push_to_hub(final_hub_repo)
processor_tuned.push_to_hub(final_hub_repo)
print(f"Final merged model and processor pushed to Hugging Face: {final_hub_repo}")
