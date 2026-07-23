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
To run training and inference efficiently without hitting GPU out-of-memory (OOM) errors, extract visual and acoustic features beforehand:

```bash
# For development/training set (CASED)
python my_utils/extract_features.py extract_whisper_audio_features CASED
python my_utils/extract_features.py extract_eva_vit_g_features CASED

# For evaluation/test set (CASED_test)
python my_utils/extract_features.py extract_whisper_audio_features CASED_test
python my_utils/extract_features.py extract_eva_vit_g_features CASED_test
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
To run an interactive prediction on a single sample:
```bash
python inference.py --cfg-path train_configs/cased_finetune.yaml --ckpt /path/to/checkpoint_best.pth
```

---

## 📖 Citation
If you find our work helpful for your research, please cite our paper as well as the base Emotion-LLaMA frameworks:

```bibtex
@inproceedings{10.1145/3776574.3832487,
  author    = {Bansal, Mohit and Hans, Arnold Sachith A and Rao, Smitha},
  title     = {MCL-SED: A Multimodal CASED-LLaMA Framework for Student Engagement Detection in Virtual Classroom},
  booktitle = {Proceedings of the International Conference on Multimodal Interaction},
  series    = {ICMI '26},
  year      = {2026},
  isbn      = {979-8-4007-2318-6},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3776574.3832487},
  location  = {Napoli, Italy},
  pages     = {1--10}
}

@inproceedings{NEURIPS2024_c7f43ada,
  author    = {Cheng, Zebang and Cheng, Zhi-Qi and He, Jun-Yan and Wang, Kai and Lin, Yuxiang and Lian, Zheng and Peng, Xiaojiang and Hauptmann, Alexander},
  title     = {Emotion-LLaMA: Multimodal Emotion Recognition and Reasoning with Instruction Tuning},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {37},
  pages     = {110805--110853},
  year      = {2024}
}

@inproceedings{10.1145/3689092.3689404,
  author    = {Cheng, Zebang and Tu, Shuyuan and Huang, Dawei and Li, Minghan and Peng, Xiaojiang and Cheng, Zhi-Qi and Hauptmann, Alexander G.},
  title     = {SZTU-CMU at MER2024: Improving Emotion-LLaMA with Conv-Attention for Multimodal Emotion Recognition},
  booktitle = {Proceedings of the 2nd International Workshop on Multimodal and Responsible Affective Computing (MRAC '24)},
  pages     = {78--87},
  year      = {2024},
  doi       = {10.1145/3689092.3689404}
}
```

---

## 📜 License
This repository is licensed under the BSD 3-Clause License. Base architecture code is adapted from MiniGPT-4 and Emotion-LLaMA.