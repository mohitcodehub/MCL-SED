import os
import re
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import cv2

from minigpt4.common.config import Config
from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from minigpt4.conversation.conversation import CONV_VISION_minigptv2

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

def parse_both_output(text):
    # Regex pattern to match Class: [0|1] and Score: [float]
    match = re.search(r"Class:\s*(\d+).*?Score:\s*([0-9.]+)", text, re.IGNORECASE)
    if match:
        cls_val = int(match.group(1))
        score_val = float(match.group(2))
        return cls_val, score_val
    
    # Fallback to separate searches
    cls_match = re.search(r"Class:\s*(\d+)", text, re.IGNORECASE)
    score_match = re.search(r"Score:\s*([0-9.]+)", text, re.IGNORECASE)
    
    cls_val = int(cls_match.group(1)) if cls_match else 0
    score_val = float(score_match.group(1)) if score_match else 3.0
    
    # Clip to valid bounds
    cls_val = min(max(cls_val, 0), 1)
    score_val = min(max(score_val, 1.0), 5.0)
    
    return cls_val, score_val

def main():
    parser = eval_parser()
    parser.add_argument("--template-csv", default="/home/VU-Senior/Downloads/CASED/submission_template.csv", help="Path to submission template CSV")
    parser.add_argument("--output-csv", default="/home/VU-Senior/Downloads/CASED/submission.csv", help="Path to save the final submission CSV")
    parser.add_argument("--test-video-dir", default="/home/VU-Senior/Downloads/CASED/evaluation_data/student_only/test", help="Path to test video folder")
    parser.add_argument("--test-features-dir", default="/home/VU-Senior/Downloads/CASED/evaluation_data/features", help="Path to test features root")
    
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
    
    # Load submission template
    template_df = pd.read_csv(args.template_csv)
    print(f"Loaded submission template with {len(template_df)} rows.")
    
    results = []
    
    # Prompt matching the CASEDDatasetFT both task prompt
    instruction_text = "Given the video and audio, determine both the classification label (0 or 1) and the continuous score (1.0 to 5.0). Output format: Class: [label], Score: [score]"
    texts = [f"<Vid><VideoHere></Vid> [both] {instruction_text} "]
    texts = [t.replace("<s>", "") for t in texts]
    
    # Keep track of outputs for verification
    num_failures = 0
    
    for idx, row in tqdm(template_df.iterrows(), total=len(template_df), desc="Running Multitask Inference"):
        video_title = row["video_title"]
        video_id = os.path.splitext(video_title)[0]
        video_path = os.path.join(args.test_video_dir, video_id + ".mp4")
        
        # 1. Load image (middle frame)
        image = extract_middle_frame(video_path)
        if image is None:
            # Fallback to zero image
            image_tensor = torch.zeros((1, 3, 224, 224)).to(device)
        else:
            image_tensor = vis_processor(image).unsqueeze(0).to(device)
            
        # 2. Load EVA-ViT-G visual feature
        video_feat_path = os.path.join(args.test_features_dir, "EVA-ViT-G", video_id + ".npy")
        if os.path.exists(video_feat_path):
            try:
                video_feature = torch.tensor(np.load(video_feat_path), dtype=torch.float32).to(device)
            except Exception as e:
                print(f"\nError loading visual feature for {video_id}: {e}")
                video_feature = torch.zeros((1, 64, 1408), dtype=torch.float32).to(device)
        else:
            video_feature = torch.zeros((1, 64, 1408), dtype=torch.float32).to(device)
            
        # 3. Load Whisper audio feature
        audio_feat_path = os.path.join(args.test_features_dir, "whisper-large-v3", video_id + ".npy")
        if os.path.exists(audio_feat_path):
            try:
                audio_feature = torch.tensor(np.load(audio_feat_path), dtype=torch.float32).to(device)
            except Exception as e:
                print(f"\nError loading audio feature for {video_id}: {e}")
                audio_feature = torch.zeros((1, 64, 1280), dtype=torch.float32).to(device)
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
                
        except Exception as e:
            print(f"\nForward pass error for {video_id}: {e}")
            cls_val, score_val = 0, 3.0
            num_failures += 1
            
        results.append({
            "video_title": video_title,
            "value": round(score_val, 2),
            "label": int(cls_val)
        })
        
        # Intermediate save every 50 steps
        if (idx + 1) % 50 == 0 or (idx + 1) == len(template_df):
            partial_df = pd.DataFrame(results)
            partial_df.to_csv(args.output_csv, index=False)
            
    print(f"\nInference completed. Total generation errors/failures: {num_failures}")
    print(f"Final submission saved to: {args.output_csv}")

if __name__ == "__main__":
    main()
