##############################
# INSTALL LIBRARIES
##############################
!pip install -q unsloth
!pip install -q peft bitsandbytes xformers
!pip install -q transformers torchvision opencv-python
!pip install huggingface_hub datasets
!pip install accelerate
!pip install trl
!pip install bitsandbytes

import os
import torch
from unsloth import FastLanguageModel
from transformers import (
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
import warnings
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')
##############################
# CHECK GPU
##############################
!nvidia-smi
#########################################
# LOGIN to huggingface [insert TOKEN]
#########################################
!huggingface-cli login --token TOKEN_HERE
#########################################
# TRAINING CONFIGURATION
#########################################
class Config:
    # Model configuration
    MODEL_NAME = "google/paligemma-3b-pt-224"
    MAX_LENGTH = 256
    BATCH_SIZE = 8
    EPOCHS = 30
    LEARNING_RATE = 5e-5
    WARMUP_STEPS = 10
    WEIGHT_DECAY = 0.01

    # LoRA configuration
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Data configuration
    DATASET_SIZE = 4000
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.2

    # Paths
    CAPTIONS_FILE = "/kaggle/input/skincap-captions/skincap_v240715.xlsx"
    OUTPUT_DIR = "./paligemma-skincap-finetuned"
    LOGS_DIR = "./logs"

print("train epoch", Config.EPOCHS)

class TQDMCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.pbar = tqdm(total=state.max_steps, desc="Training")

    def on_step_end(self, args, state, control, **kwargs):
        self.pbar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        self.pbar.close()

# Dataset class
class SkinCAPDataset(torch.utils.data.Dataset):
    def __init__(self, images, captions):
        self.images = images
        self.captions = captions

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self._get_single_item(i) for i in idx]
        else:
            return self._get_single_item(idx)

    def _get_single_item(self, idx):
        image = self.images[idx]
        caption = str(self.captions[idx])

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return {
            'image': image,
            'caption': caption
        }
#########################################
# DATA COLLATOR
#########################################
def data_collator(batch):
    if isinstance(batch[0], list):
        batch = [item for sublist in batch for item in sublist]

    images = [item['image'] for item in batch]
    captions = [item['caption'] for item in batch]
    prompts = ["<image>caption en"] * len(batch)

    processor = AutoProcessor.from_pretrained(Config.MODEL_NAME)
    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        max_length=Config.MAX_LENGTH,
        padding="max_length",
        truncation=True
    )

    inputs['labels'] = processor.tokenizer(
        captions,
        return_tensors="pt",
        max_length=Config.MAX_LENGTH,
        padding="max_length",
        truncation=True
    )['input_ids']

    return inputs
#########################################
# METRICS
#########################################
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predicted_ids = np.argmax(predictions, axis=-1)
    decoded_preds = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
    decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

    accuracy = accuracy_score(decoded_labels, decoded_preds)
    return {"accuracy": accuracy}
#########################################
# TRAINING
#########################################
# Training arguments
def create_training_arguments():
    return TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=2,
        warmup_steps=Config.WARMUP_STEPS,
        weight_decay=Config.WEIGHT_DECAY,
        learning_rate=Config.LEARNING_RATE,
        logging_dir=Config.LOGS_DIR,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        fp16=True,
        remove_unused_columns=False,
        report_to="none"
    )


def generate_sample_predictions(model, processor, val_dataset, num_samples=5):
    print(f"\n🔍 Generating {num_samples} sample predictions...")

    model.eval()
    device = next(model.parameters()).device

    for i in tqdm(range(min(num_samples, len(val_dataset))), desc="Generating predictions"):
        item = val_dataset[i]

        image = item['image']
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        prompt = "<image>caption en"  # ADDED <image> TOKEN
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            max_length=Config.MAX_LENGTH,
            padding="max_length",
            truncation=True
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=processor.tokenizer.pad_token_id
            )

        generated_text = processor.decode(outputs[0], skip_special_tokens=True)
        # Adjust split since we added <image> token
        predicted_caption = generated_text.split("caption en\n")[-1].strip()

        print(f"\nSample {i+1}:")
        print(f"Ground Truth: {item['caption']}")
        print(f"Predicted:    {predicted_caption}")
        print("-" * 50)

# Main training function
def train_model():
    print("Starting model training...")

    # Load dataset with images
    hf_dataset = load_dataset("joshuachou/SkinCAP", split="train")
    hf_dataset = hf_dataset.select(range(Config.DATASET_SIZE))

    # Load captions from Excel
    captions_df = pd.read_excel(Config.CAPTIONS_FILE, header=1)
    captions_df = captions_df.iloc[:Config.DATASET_SIZE]

    # Split dataset
    train_size = int(Config.DATASET_SIZE * Config.TRAIN_SPLIT)
    train_images = hf_dataset["image"][:train_size]
    train_captions = captions_df["caption_en"].tolist()[:train_size]
    val_images = hf_dataset["image"][train_size:]
    val_captions = captions_df["caption_en"].tolist()[train_size:]

    # Create datasets
    train_dataset = SkinCAPDataset(train_images, train_captions)
    val_dataset = SkinCAPDataset(val_images, val_captions)

    # Load processor
    processor = AutoProcessor.from_pretrained(Config.MODEL_NAME)

    # Quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # Load model
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        Config.MODEL_NAME,
        quantization_config=quantization_config,
        torch_dtype=torch.float16
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        target_modules=Config.LORA_TARGET_MODULES,
        lora_dropout=Config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Training arguments
    training_args = create_training_arguments()

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), TQDMCallback()]
    )

    # Train model
    trainer.train()

    # Save model
    model.save_pretrained(Config.OUTPUT_DIR)
    processor.save_pretrained(Config.OUTPUT_DIR)

    return model, processor, val_dataset
#########################################
# LOAD DATASET
#########################################
hf_dataset = load_dataset("joshuachou/SkinCAP", split="train")
hf_dataset = hf_dataset.select(range(Config.DATASET_SIZE))

captions_df = pd.read_excel(Config.CAPTIONS_FILE, header=1)
captions_df = captions_df.iloc[:Config.DATASET_SIZE]

train_size = int(Config.DATASET_SIZE * Config.TRAIN_SPLIT)
train_images = hf_dataset["image"][:train_size]
train_captions = captions_df["caption_en"].tolist()[:train_size]
val_images = hf_dataset["image"][train_size:]
val_captions = captions_df["caption_en"].tolist()[train_size:]

train_dataset = SkinCAPDataset(train_images, train_captions)
val_dataset = SkinCAPDataset(val_images, val_captions)
#########################################
# LOAD PROCESSOR + MODEL
#########################################
processor = AutoProcessor.from_pretrained(Config.MODEL_NAME)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = PaliGemmaForConditionalGeneration.from_pretrained(
    Config.MODEL_NAME,
    quantization_config=quantization_config,
    torch_dtype=torch.float16
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=Config.LORA_R,
    lora_alpha=Config.LORA_ALPHA,
    target_modules=Config.LORA_TARGET_MODULES,
    lora_dropout=Config.LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
#########################################
# TRAINING
#########################################
training_args = create_training_arguments()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3), TQDMCallback()]
)

trainer.train()
#########################################
# SAVE LOCALLY
#########################################
model.save_pretrained(Config.OUTPUT_DIR)
processor.save_pretrained(Config.OUTPUT_DIR)
#########################################
# GENERATE SAMPLE PREDICTIONS
#########################################
generate_sample_predictions(model, processor, val_dataset)
#########################################
# PUSH TO HUB
#########################################
model.push_to_hub("finetuned_paligemma_3b_30_epoch_skincap")
processor.push_to_hub("finetuned_paligemma_3b_30_epoch_skincap")
print("Model and processor saved to huggingface")
