# MCL-SED: A Multimodal CASED-LLaMA Framework for Student Engagement Detection in Virtual Classroom

## 🌟 Overview
**MCL-SED** is a state-of-the-art multimodal framework designed for automated **Student Engagement Detection (SED)** in virtual learning environments. Built on top of Parameter-Efficient Fine-Tuning (PEFT) paradigms and multimodal alignment backbones, MCL-SED integrates visual representations, vocal tones, and transcription subtitles to predict student engagement levels.

Our framework secured **🥈 2nd Place globally** in the official **ACM ICMI 2026 Context-Aware Student Engagement Detection (CASED) Grand Challenge**, proving its capability to generalize under strict participant-disjoint evaluation protocols.

---

## 🏗️ Architecture
MCL-SED extracts and fuses rich, heterogeneous representations of student behavior across three primary modalities:
1. **Visual Cues**: Pre-extracted spatiotemporal features using **EVA-ViT-G**.
2. **Acoustic Cues**: Pre-extracted audio embeddings from **Whisper-Large-v3**.
3. **Textual Subtitles**: Subtitle text transcripts aligned with video timestamps.
4. **Large Language Model (LLM)**: A LoRA-adapted **LLaMA-2-7B-chat-hf** model acting as the central multi-task reasoning engine.

---

## ⚙️ Installation & Setup

### 1. Environment Configuration
Create a dedicated `conda` environment and install all dependencies:
```bash
conda create --name mcl-sed python=3.10.16 -y
conda activate mcl-sed
pip install -r requirement.txt
```

### 2. General Checkpoints & Models
Download the following baseline checkpoints and place them in your model workspace:
* **Audio Encoder**: [Whisper-Large-v3](https://huggingface.co/openai/whisper-large-v3)
* **Visual Encoder**: [EVA-ViT-G](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth)
* **Language Model Backbone**: [LLaMA-2-7B-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
* **Base Multimodal Projector**: [MiniGPT-v2 Checkpoint](https://github.com/Vision-CAIR/MiniGPT-4)

---

## 📊 Feature Extraction
To run training efficiently without hitting out-of-memory (OOM) errors, extract temporal and visual features beforehand:

```bash
# Extract acoustic features using Whisper-Large-v3
python my_utils/extract_features.py extract_whisper_audio_features [dataset_name]

# Extract spatiotemporal features using EVA-ViT-G
python my_utils/extract_features.py extract_eva_vit_g_features [dataset_name]
```

---

## 🏋️ Training & Fine-Tuning

1. Configure your dataset paths in `minigpt4/configs/datasets/default.yaml`.
2. Review the model setup paths in the training configuration: `train_configs/cased_finetune.yaml`.
3. Launch the fine-tuning training job (using distributed multi-GPU setup if available):

```bash
# Distributed fine-tuning across 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 train.py --cfg-path train_configs/cased_finetune.yaml
```

---

## 🔍 Inference & Evaluation

### 1. Compute Validation Metrics
To evaluate model performance (F1, MCC, CCC, RMSE, etc.) on the CASED validation split:
```bash
export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=0 python evaluate_llama_metrics.py \
  --cfg-path train_configs/cased_finetune.yaml \
  --ckpt /path/to/checkpoint_best.pth \
  --eval-csv /path/to/val_split.csv \
  --video-dir /path/to/validation_videos \
  --features-dir /path/to/extracted_features
```

### 2. Generate Leaderboard Submission
To run inference on the test set and generate a submission CSV:
```bash
export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=0 python run_inference_cased.py \
  --cfg-path train_configs/cased_finetune.yaml \
  --ckpt /path/to/checkpoint_best.pth \
  --template-csv /path/to/submission_template.csv \
  --output-csv /path/to/save/submission.csv \
  --test-video-dir /path/to/test_videos \
  --test-features-dir /path/to/extracted_features
```

### 3. Single-Sample Inference
To run a prediction on a single test sample:
```bash
python inference.py
```

---

## 📖 Citation
If you find our work helpful for your research, please cite our paper:
```bibtex
@inproceedings{mcl_sed_icmi2026,
  author    = {Mohit Bansal and Arnold Sachith A Hans and Smitha Rao},
  title     = {MCL-SED: A Multimodal CASED-LLaMA Framework for Student Engagement Detection in Virtual Classroom},
  booktitle = {Proceedings of the ACM International Conference on Multimodal Interaction (ICMI)},
  year      = {2026}
}
```

---

## 📜 License
This repository is licensed under the BSD 3-Clause License. Base architecture code is adapted from MiniGPT-4 and Emotion-LLaMA.