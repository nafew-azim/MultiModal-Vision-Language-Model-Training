##############################
# INSTALL LIBRARIES
##############################
!pip install git+https://github.com/huggingface/transformers.git@main
!pip install -q datasets
!pip install rouge-score
!pip install --upgrade datasets
!pip install huggingface_hub
!huggingface-cli login --token TOKEN_HERE

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
import nltk
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score
import torch.nn.functional as F
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoImageProcessor, ViTModel
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoProcessor, BlipForConditionalGeneration, AutoTokenizer
import time
import random
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
from nltk.translate.meteor_score import meteor_score
import os
from huggingface_hub import Repository

warnings.filterwarnings("ignore")

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset and captions
dataset = load_dataset("eltorio/ROCOv2-radiology")

# Assign each split to a variable
train_dataset = dataset["train"]
val_dataset   = dataset["validation"]
test_dataset  = dataset["test"]

# Print out the number of examples in each
print(f"Train set size:      {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size:       {len(test_dataset)}")

# Define custom dataset class with truncation enabled
class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Enable truncation to ensure all sequences are <= max_length
        encoding = self.processor(images=item["image"], text=item["caption"], padding="max_length", truncation=True, return_tensors="pt")
        # Remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        return encoding

# Initialize processor and model
#processor_tuned = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#model_tuned = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load model and processor from checkpoint
processor_tuned = AutoProcessor.from_pretrained('/kaggle/input/new_model_01/pytorch/default/1/checkpoints/epoch_1')
model_tuned = BlipForConditionalGeneration.from_pretrained('/kaggle/input/new_model_01/pytorch/default/1/checkpoints/epoch_1')

model_tuned.to(device)

# Create datasets and dataloaders
train_dataset = ImageCaptioningDataset(train_dataset, processor_tuned)
val_dataset = ImageCaptioningDataset(val_dataset, processor_tuned)
test_dataset = ImageCaptioningDataset(test_dataset, processor_tuned)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=2)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=2)

# Training setup
optimizer = torch.optim.AdamW(model_tuned.parameters(), lr=2e-5)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_tuned.to(device)

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

# Create checkpoint directory
checkpoint_dir = "/kaggle/working/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Training loop with checkpoint saving and pushing to Hugging Face Hub
num_epochs = 4
for epoch in range(num_epochs):
    model_tuned.train()
    for idx, batch in enumerate(train_dataloader):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        outputs = model_tuned(pixel_values=pixel_values, input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        print(f"Epoch {epoch}, Batch {idx}, Training Loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    val_loss = compute_val_loss(model_tuned, val_dataloader, device)
    print(f"Epoch {epoch}, Validation Loss: {val_loss}")
    scheduler.step(val_loss)

    # Save checkpoint
    epoch_dir = os.path.join(checkpoint_dir, f"epoch_{epoch+1}")
    os.makedirs(epoch_dir, exist_ok=True)
    model_tuned.save_pretrained(epoch_dir)
    processor_tuned.save_pretrained(epoch_dir)
    print(f"Checkpoint saved for epoch {epoch+1} at {epoch_dir}")

    # Push checkpoint to Hugging Face Hub
    repo_id = f"blip-image-captioning-base-roco-epoch-{epoch+1}"
    repo = Repository(local_dir=epoch_dir, repo_id=repo_id)
    repo.push_to_hub(commit_message=f"Add checkpoint for epoch {epoch+1}")
    print(f"Checkpoint for epoch {epoch+1} pushed to Hugging Face: {repo_id}")

# Evaluate on test set
test_loss = compute_test_loss(model_tuned, test_dataloader, device)
print(f"Test Loss: {test_loss}")

# Save final model and processor
final_dir = "/kaggle/working/finetuned_blip_image_captioning_roco_5_epoch"
os.makedirs(final_dir, exist_ok=True)
model_tuned.save_pretrained(final_dir)
processor_tuned.save_pretrained(final_dir)
print(f"Final model and processor saved to {final_dir}")

# Push final model to Hugging Face Hub
final_repo_id = "blip-image-captioning-base-roco-final_5_epoch"
repo = Repository(local_dir=final_dir, repo_id=final_repo_id)
repo.push_to_hub(commit_message="Add final model")
print(f"Final model and processor pushed to Hugging Face: {final_repo_id}")
