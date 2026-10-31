import os
import glob
import zipfile
import pandas as pd
from pathlib import Path

# Dynamically resolve the project root directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Define output directory for the processed dataset
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "clean"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define window size: 5 seconds of data at 100 Hz = 500 rows per chunk
CHUNK_SIZE = 500 

def process_and_chunk_zips(data_folder: Path):
    zip_files = list(data_folder.glob("*.zip"))
    chunk_counts = {"still": 0, "standing": 0, "walking": 0, "jumping": 0}
    
    for zip_path in zip_files:
        # Extract the activity label from the filename (e.g., 'walking-2026-10-30.zip' -> 'walking')
        activity = os.path.basename(zip_path).split('-')[0] 
        
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Load CSVs and ensure chronological order
            acc = pd.read_csv(z.open('Accelerometer.csv')).sort_values('seconds_elapsed')
            acc_uncal = pd.read_csv(z.open('AccelerometerUncalibrated.csv')).sort_values('seconds_elapsed')
            gyro = pd.read_csv(z.open('Gyroscope.csv')).sort_values('seconds_elapsed')
            
            # Reorder columns from Sensor Logger's default (z,y,x) to standard (x,y,z) 
            # and rename to prevent overlap during merging
            acc = acc[['seconds_elapsed', 'x', 'y', 'z']].rename(columns={'x': 'acc_x', 'y': 'acc_y', 'z': 'acc_z'})
            gyro = gyro[['seconds_elapsed', 'x', 'y', 'z']].rename(columns={'x': 'gyr_x', 'y': 'gyr_y', 'z': 'gyr_z'})
            acc_uncal = acc_uncal[['seconds_elapsed', 'x', 'y', 'z']].rename(columns={'x': 'uncal_x', 'y': 'uncal_y', 'z': 'uncal_z'})
            
            # Merge calibrated sensors (they share exact nanosecond timestamps)
            merged = pd.merge(acc, gyro, on='seconds_elapsed', how='inner')
            
            # Align uncalibrated accelerometer data using nearest time elapsed
            # This handles the slight hardware offset in the internal clocks
            final_df = pd.merge_asof(merged, acc_uncal, on='seconds_elapsed', direction='nearest')
            
            # Slice the continuous recording into fixed 5-second chunks
            num_chunks = len(final_df) // CHUNK_SIZE
            
            for i in range(num_chunks):
                # Isolate the 500-row window
                chunk = final_df.iloc[i*CHUNK_SIZE : (i+1)*CHUNK_SIZE]
                
                # Save as an independent CSV file to build out the dataset
                chunk_id = chunk_counts[activity]
                filename = OUTPUT_DIR / f"{activity}_chunk_{chunk_id:03d}.csv"
                chunk.to_csv(filename, index=False)
                
                chunk_counts[activity] += 1

    print("Data extraction and chunking complete. Total files generated:")
    print(chunk_counts)

# Run the preprocessing pipeline
if __name__ == "__main__":
    process_and_chunk_zips(DATA_DIR)