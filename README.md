# MultiModal-Vision-Language-Model-Training

This repository contains the code and resources for fine-tuning various **multimodal vision-language models** on medical imaging datasets, specifically for **image captioning tasks**.  
The project leverages state-of-the-art models to generate accurate and descriptive captions for medical images, with applications in **dermatology (SkinCAP dataset)** and **radiology (ROCOv2 dataset)**.  
It includes training/evaluation scripts for models such as **PaliGemma, BLIP-2, BLIP, SmolVLM, Qwen-VL, and Florence-2**, optimized with **quantization** and **LoRA (Low-Rank Adaptation)**.

---

## 📑 Table of Contents
- [Overview](#overview)  
- [Features](#features)  
- [Datasets](#datasets)  
- [Models](#models)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Repository Structure](#repository-structure)  
- [Training Details](#training-details)  
- [Evaluation](#evaluation)  
- [Contributing](#contributing)  
- [License](#license)  
- [Acknowledgments](#acknowledgments)  

---

## 🔍 Overview
The goal of this project is to **fine-tune multimodal vision-language models** to generate high-quality captions for medical images.  

- Includes training & evaluation scripts for **SkinCAP (dermatology)** and **ROCOv2 (radiology)** datasets.  
- Supports **LoRA, 4-bit quantization, and custom preprocessing pipelines**.  
- Models are saved locally and pushed to the **Hugging Face Hub** for reproducibility and sharing.  

---

## ✨ Features
- **Multimodal Model Training:** Fine-tuning of multiple vision-language models: PaliGemma, BLIP-2, BLIP, SmolVLM, Qwen-VL, and Florence-2.  
- **Efficient Fine-Tuning:** LoRA + 4-bit quantization for reduced memory usage, enabling training on consumer GPUs.  
- **Custom Dataset Classes:** Preprocessing pipelines for medical imaging data.  
- **Comprehensive Evaluation:** Accuracy, ROUGE, BLEU, and METEOR metrics.  
- **Hugging Face Integration:** Push models and processors to Hugging Face Hub.  
- **Checkpointing:** Regular checkpoint saving for resuming and reproducibility.  

---

## 📊 Datasets
1. **SkinCAP**  
   - Dermatology dataset with images + captions.  
   - Available via Hugging Face: [`joshuachou/SkinCAP`](https://huggingface.co/datasets/joshuachou/SkinCAP).  
   - Captions provided in [`skincap_v240715.xlsx`](https://www.kaggle.com/datasets/nafewazim/skincap-captions).  
   - Subset of **4,000 samples** used for training/evaluation.  

2. **ROCOv2**  
   - Radiology dataset with image-caption pairs.  
   - Available via Hugging Face: [`eltorio/ROCOv2-radiology`](https://huggingface.co/datasets/eltorio/ROCOv2-radiology).  
   - Split into **train/validation/test** sets.  

---

## 🤖 Models
- **PaliGemma-3B** → `paligemma_3b_skincap.py`  
- **BLIP-2 OPT-2.7B** → `blip_2_opt_2_7b_roco_v2.py`  
- **BLIP Base** → `blip_base_roco_v2.py`  
- **SmolVLM-500M** → `smolvlm_roco_v2.py`  
- **Qwen-VL-2B (Unsloth)** → `qwen_vl_2b_unsloth_skincap.py`, `qwen_vl_2b_unsloth_roco_v2.py`  
- **Florence-2** → `florence_2.py`  

Each script includes **data loading, preprocessing, training, evaluation, and Hugging Face Hub integration**.

---

## ⚙️ Installation
```bash
# Clone repo
git clone https://github.com/nafew-azim/MultiModal-Vision-Language-Model-Training.git
cd MultiModal-Vision-Language-Model-Training

# Create environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
````

### Hugging Face Login

```bash
huggingface-cli login --token YOUR_HF_TOKEN
```

---

## 🚀 Usage

Each script can be run independently:

```bash
python paligemma_3b_skincap.py
# or
python blip_2_opt_2_7b_roco_v2.py
```

**Steps:**

1. Update dataset paths in the script (e.g., `CAPTIONS_FILE`).
2. Replace `TOKEN_HERE` with your Hugging Face token.
3. Run training → checkpoints saved locally + pushed to Hugging Face Hub.
4. Use provided functions to generate predictions.

✅ Example: Training **PaliGemma on SkinCAP**

```bash
python paligemma_3b_skincap.py
```

---

## 📂 Repository Structure

```
MultiModal-Vision-Language-Model-Training/
├── paligemma_3b_skincap.py        # Fine-tune PaliGemma on SkinCAP
├── blip_2_opt_2_7b_roco_v2.py     # Fine-tune BLIP-2 OPT-2.7B on ROCOv2
├── blip_base_roco_v2.py           # Fine-tune BLIP Base on ROCOv2
├── smolvlm_roco_v2.py             # Fine-tune SmolVLM on ROCOv2
├── qwen_vl_2b_unsloth_skincap.py # Fine-tune Qwen-VL-2B on SkinCAP
├── qwen_vl_2b_unsloth_roco_v2.py  # Fine-tune Qwen-VL-2B on ROCOv2
├── florence_2.py                  # Fine-tune Florence-2 on SkinCAP
├── requirements.txt               # Dependencies
├── README.md                      # Project documentation
```

---

## 🏋️ Training Details

* **Optimizer:** Adam / AdamW, lr = `2e-5` → `5e-5`
* **LR Scheduler:** ReduceLROnPlateau
* **Quantization:** 4-bit with `BitsAndBytesConfig`
* **LoRA:** Efficient fine-tuning of subset parameters
* **Checkpoints:** Saved after each epoch + optional push to Hugging Face Hub
* **Epochs:** 4 → 30 (varies per script)

---

## 📈 Evaluation

* **Losses:** Train/val/test monitoring
* **Metrics:** Accuracy, ROUGE, BLEU, METEOR
* **Qualitative:** Sample caption generation on validation set

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo
2. Create a branch (`feature/new-feature`)
3. Submit a PR with a detailed description

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

* [Hugging Face](https://huggingface.co/) → Transformers, Datasets, Model Hub
* [Unsloth](https://github.com/unslothai/unsloth) → Optimized VLM training
* [Kaggle](https://www.kaggle.com/) → SkinCAP captions dataset
* Google Colab → Development environment

---

📩 For questions or issues, open an **[issue](../../issues)** or contact: **[nafew.azim@gmail.com](mailto:nafew.azim@gmail.com)**
