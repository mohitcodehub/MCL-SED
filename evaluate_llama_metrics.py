import os
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import cv2

from minigpt4.common.config import Config
from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, mean_absolute_error, mean_squared_error, r2_score
)
from scipy.stats import pearsonr

def concordance_cc(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covar = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    numerator = 2 * covar
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denominator == 0:
        return 1.0
    return numerator / denominator

def extract_middle_frame(video_path):
    try:
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            return None
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return None
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        success, frame = video_capture.read()
        video_capture.release()
        if success and frame is not None:
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"Error extracting middle frame from {video_path}: {e}")
    return None

def main():
    parser = eval_parser()
    parser.add_argument("--eval-csv", required=True, help="Path to split CSV file (e.g., val_split.csv)")
    parser.add_argument("--video-dir", default="/home/VU-Senior/Downloads/CASED_Main/development_data/student_only/train", help="Path to video folder")
    parser.add_argument("--features-dir", default="/home/VU-Senior/Downloads/CASED_Main/development_data/features", help="Path to features root")
    
    args = parser.parse_args()
    
    # Force the config to use the provided checkpoint by injecting it into options
    if args.ckpt:
        if args.options is None:
            args.options = []
        args.options.append(f"model.ckpt={args.ckpt}")
        
    # Initialize the model and vis processor
    model, vis_processor = init_model(args)
    model.eval()
    
    device = next(model.parameters()).device
    print(f"Model successfully loaded and running on device: {device}")
    
    df = pd.read_csv(args.eval_csv)
    print(f"Loaded CSV with {len(df)} rows.")
    
    y_true_cls = []
    y_pred_cls = []
    y_true_reg = []
    y_pred_reg = []
    
    instruction_text = "Given the video and audio, determine both the classification label (0 or 1) and the continuous score (1.0 to 5.0). Output format: Class: [label], Score: [score]"
    texts = [f"<Vid><VideoHere></Vid> [both] {instruction_text} "]
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        video_title = row["video_title"]
        video_id = os.path.splitext(video_title)[0]
        video_path = os.path.join(args.video_dir, video_title)
        
        # 1. Load image (middle frame)
        image = extract_middle_frame(video_path)
        if image is None:
            image_tensor = torch.zeros((1, 3, 224, 224)).to(device)
        else:
            image_tensor = vis_processor(image).unsqueeze(0).to(device)
            
        # 2. Load EVA-ViT-G visual feature
        video_feat_path = os.path.join(args.features_dir, "EVA-ViT-G", video_id + ".npy")
        if os.path.exists(video_feat_path):
            video_feature = torch.tensor(np.load(video_feat_path), dtype=torch.float32).to(device)
        else:
            video_feature = torch.zeros((1, 64, 1408), dtype=torch.float32).to(device)
            
        # 3. Load Whisper audio feature
        audio_feat_path = os.path.join(args.features_dir, "whisper-large-v3", video_id + ".npy")
        if os.path.exists(audio_feat_path):
            audio_feature = torch.tensor(np.load(audio_feat_path), dtype=torch.float32).to(device)
        else:
            audio_feature = torch.zeros((1, 64, 1280), dtype=torch.float32).to(device)
            
        # 4. Generate
        try:
            with torch.no_grad():
                with model.maybe_autocast():
                    samples = {
                        "image": image_tensor,
                        "video_features": video_feature.unsqueeze(0),
                        "audio_features": audio_feature.unsqueeze(0),
                        "instruction_input": texts
                    }
                    outputs = model(samples)
                    
                cls_val = torch.argmax(outputs["class_logits"], dim=-1).item()
                score_val = outputs["score_preds"].item()
                score_val = max(1.0, min(5.0, score_val)) # clamp to valid range
                
                y_true_cls.append(int(row["label"]))
                y_pred_cls.append(int(cls_val))
                y_true_reg.append(float(row["value"]))
                y_pred_reg.append(float(score_val))
        except Exception as e:
            print(f"\nForward pass error for {video_id}: {e}")
            
    # Calculate classification metrics
    acc = accuracy_score(y_true_cls, y_pred_cls)
    f1_macro = f1_score(y_true_cls, y_pred_cls, average="macro")
    f1_weighted = f1_score(y_true_cls, y_pred_cls, average="weighted")
    prec_macro = precision_score(y_true_cls, y_pred_cls, average="macro", zero_division=0)
    rec_macro = recall_score(y_true_cls, y_pred_cls, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true_cls, y_pred_cls)
    
    # Calculate regression metrics
    ccc = concordance_cc(y_true_reg, y_pred_reg)
    pearson_r, _ = pearsonr(y_true_reg, y_pred_reg)
    r2 = r2_score(y_true_reg, y_pred_reg)
    mae = mean_absolute_error(y_true_reg, y_pred_reg)
    mse = mean_squared_error(y_true_reg, y_pred_reg)
    rmse = np.sqrt(mse)
    
    print("\n" + "="*50)
    print(f"Evaluation Results for {args.eval_csv}")
    print("="*50)
    print("Classification Metrics:")
    print(f"  Accuracy:         {acc:.4f}")
    print(f"  F1 Macro:         {f1_macro:.4f}")
    print(f"  F1 Weighted:      {f1_weighted:.4f}")
    print(f"  Precision Macro:  {prec_macro:.4f}")
    print(f"  Recall Macro:     {rec_macro:.4f}")
    print(f"  MCC:              {mcc:.4f}")
    print("-"*50)
    print("Regression Metrics:")
    print(f"  CCC:              {ccc:.4f}")
    print(f"  Pearson:          {pearson_r:.4f}")
    print(f"  R2:               {r2:.4f}")
    print(f"  MAE:              {mae:.4f}")
    print(f"  MSE:              {mse:.4f}")
    print(f"  RMSE:             {rmse:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
