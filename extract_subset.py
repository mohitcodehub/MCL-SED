import os
import pandas as pd
from my_utils.extract_features import FeatureExtractor
import my_utils.config as config

def main():
    print("Initializing FeatureExtractor...")
    fe = FeatureExtractor()
    
    # Load splits
    val_csv = "/home/VU-Senior/Downloads/CASED/development_data/val_split.csv"
    val_df = pd.read_csv(val_csv)
    
    # Let's extract features for the first 10 validation files
    video_titles = val_df['video_title'].head(10).tolist()
    print(f"Extracting features for a subset of {len(video_titles)} videos...")
    
    video_root = config.PATH_TO_RAW_VIDEO['CASED']
    features_root = config.PATH_TO_FEATURES['CASED']
    
    whisper_save_dir = os.path.join(features_root, "whisper-large-v3")
    eva_save_dir = os.path.join(features_root, "EVA-ViT-G")
    
    os.makedirs(whisper_save_dir, exist_ok=True)
    os.makedirs(eva_save_dir, exist_ok=True)
    
    for i, title in enumerate(video_titles):
        video_path = os.path.join(video_root, title)
        name_only = os.path.splitext(title)[0]
        
        whisper_npy = os.path.join(whisper_save_dir, name_only + ".npy")
        eva_npy = os.path.join(eva_save_dir, name_only + ".npy")
        
        print(f"[{i+1}/{len(video_titles)}] Processing {title}...")
        
        if not os.path.exists(whisper_npy):
            print("  Extracting Whisper features...")
            fe._extract_single_whisper_feature(video_path, whisper_npy)
        else:
            print("  Whisper features already exist.")
            
        if not os.path.exists(eva_npy):
            print("  Extracting EVA features...")
            fe._extract_single_eva_feature(video_path, eva_npy)
        else:
            print("  EVA features already exist.")
            
    print("Subset feature extraction finished successfully!")

if __name__ == '__main__':
    main()
