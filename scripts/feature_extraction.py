from pathlib import Path
import pandas as pd
import numpy as np
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler

# 1. Dynamically resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_DIR = PROJECT_ROOT / "data" / "clean"
OUTPUT_FILE = PROJECT_ROOT / "data" / "extracted_features.csv"

def extract_features_from_window(df):
    """
    Extracts time and frequency domain features from a 5-second window of sensor data.
    """
    features = {}
    
    # Calculate Signal Magnitude Area (Vector Magnitude) for the calibrated accelerometer
    # This combines x, y, and z into a single overall movement intensity metric
    acc_mag = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    
    # --- TIME DOMAIN FEATURES ---
    # Variance of Acceleration: Captures the spread/intensity of the movement (Jumping vs Standing)
    features['acc_mag_var'] = np.var(acc_mag)
    
    # RMS of Acceleration: Captures the average power of the signal
    features['acc_mag_rms'] = np.sqrt(np.mean(acc_mag**2))
    
    # Posture (Mean of Uncalibrated Z): Differentiates 'Still' (flat on table, ~ -1g) 
    # from 'Standing' (upright in pocket, ~ 0g on Z-axis depending on orientation)
    features['uncal_z_mean'] = np.mean(df['uncal_z'])
    
    # Variance of Gyroscope Y-axis: Captures forward swing rotation (highly active in walking)
    features['gyr_y_var'] = np.var(df['gyr_y'])
    
    # --- FREQUENCY DOMAIN FEATURES ---
    # Dominant Frequency via FFT: Finds the primary rhythm (e.g., step rate during walking)
    
    # Perform Fast Fourier Transform (subtract mean to center the signal around zero)
    fft_vals = np.abs(fft(acc_mag.values - np.mean(acc_mag.values)))
    
    # Take only the positive frequencies (first half)
    fft_vals = fft_vals[1:len(fft_vals)//2] 
    
    # 100Hz sampling rate means time step (d) is 0.01 seconds
    freqs = np.fft.fftfreq(len(acc_mag), d=0.01)[1:len(fft_vals)+1] 
    
    # Identify the frequency with the highest amplitude (ignoring baseline drift below 0.5 Hz)
    valid_idx = np.where(freqs > 0.5)[0] 
    best_valid_idx = valid_idx[np.argmax(fft_vals[valid_idx])]
    
    features['dominant_freq'] = freqs[best_valid_idx]
    
    return features

def build_feature_dataset(input_folder: Path):
    dataset = []
    labels = []
    
    # Find all the chunked CSVs generated from Phase 1
    chunk_files = list(input_folder.glob("*.csv"))
    
    for file_path in chunk_files:
        # Extract label from filename (e.g., 'walking_chunk_001.csv' -> 'walking')
        activity_label = file_path.name.split('_')[0]
        
        # Read the 5-second chunk
        df = pd.read_csv(file_path)
        
        # Compute features
        feat_dict = extract_features_from_window(df)
        
        # Append to our dataset lists
        dataset.append(feat_dict)
        labels.append(activity_label)
        
    # Convert to DataFrame
    features_df = pd.DataFrame(dataset)
    
    # --- NORMALIZATION ---
    # Z-Score Standardization (Mean=0, Std=1)
    # This prevents features with naturally larger values (like variance) from overpowering small values (like frequencies)
    scaler = StandardScaler()
    feature_columns = features_df.columns
    
    normalized_data = scaler.fit_transform(features_df)
    final_df = pd.DataFrame(normalized_data, columns=feature_columns)
    
    # Append the labels back to the normalized dataframe
    final_df['label'] = labels
    
    return final_df

if __name__ == "__main__":
    # Generate the normalized feature set
    final_dataset = build_feature_dataset(INPUT_DIR)
    
    # Save to a master CSV file for the HMM
    final_dataset.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Feature extraction complete! Dataset shape: {final_dataset.shape}")
    print(f"Saved to: {OUTPUT_FILE}")