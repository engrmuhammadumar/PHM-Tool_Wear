import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path

def explore_brats_data(data_path, metadata_path):
    """
    Explore BraTS2020 dataset structure and characteristics
    """
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    print("Dataset Overview:")
    print(f"Total slices: {len(metadata)}")
    print(f"Unique volumes: {metadata['volume'].nunique()}")
    print(f"Slices per volume: {len(metadata) // metadata['volume'].nunique()}")
    
    # Analyze class distribution
    print("\nClass Distribution:")
    print(f"Background-only slices (target=0): {sum(metadata['target'] == 0)}")
    print(f"Tumor-containing slices (target=1): {sum(metadata['target'] == 1)}")
    
    # Background ratio analysis
    print(f"\nBackground Ratio Statistics:")
    print(metadata['background_ratio'].describe())
    
    # Sample data loading
    sample_file = metadata.iloc[0]['slice_path']
    print(f"\nLoading sample file: {sample_file}")
    
    try:
        with h5py.File(os.path.join(data_path, sample_file.split('/')[-1]), 'r') as f:
            print("HDF5 file contents:")
            for key in f.keys():
                print(f"  {key}: shape {f[key].shape}, dtype {f[key].dtype}")
                
    except FileNotFoundError:
        print("Sample file not found. Check your data path.")
    
    return metadata

def visualize_sample_data(data_path, metadata, volume_id=41):
    """
    Visualize sample slices from a specific volume
    """
    volume_data = metadata[metadata['volume'] == volume_id]
    
    # Select slices with different tumor presence
    background_slice = volume_data[volume_data['target'] == 0].iloc[0]
    tumor_slice = volume_data[volume_data['target'] == 1].iloc[0]
    
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    
    for idx, (slice_info, title) in enumerate([(background_slice, "Background"), 
                                               (tumor_slice, "With Tumor")]):
        try:
            slice_file = os.path.join(data_path, slice_info['slice_path'].split('/')[-1])
            with h5py.File(slice_file, 'r') as f:
                # Assuming the structure has different MRI modalities
                for modality_idx, key in enumerate(list(f.keys())[:4]):
                    axes[idx, modality_idx].imshow(f[key][:], cmap='gray')
                    axes[idx, modality_idx].set_title(f"{title} - {key}")
                    axes[idx, modality_idx].axis('off')
        except Exception as e:
            print(f"Error loading slice: {e}")
    
    plt.tight_layout()
    plt.show()

def analyze_noise_characteristics(metadata):
    """
    Analyze noise characteristics across different slices
    """
    # Group by background ratio to understand noise distribution
    noise_analysis = metadata.groupby(pd.cut(metadata['background_ratio'], bins=10)).agg({
        'label0_pxl_cnt': ['mean', 'std'],
        'label1_pxl_cnt': ['mean', 'std'],
        'label2_pxl_cnt': ['mean', 'std']
    })
    
    print("Noise Analysis by Background Ratio:")
    print(noise_analysis)
    
    return noise_analysis

# Usage example
if __name__ == "__main__":
    # Update these paths to match your downloaded data
    data_path = r"E:\Collaboration Work\With Faisal\MS\data\BraTS2020_training_data\content\data"
    metadata_path = r"E:\Collaboration Work\With Faisal\MS\data\BraTS20 Training Metadata.csv"
    
    # Explore dataset
    metadata = explore_brats_data(data_path, metadata_path)
    
    # Visualize samples
    visualize_sample_data(data_path, metadata)
    
    # Analyze noise
    noise_analysis = analyze_noise_characteristics(metadata)