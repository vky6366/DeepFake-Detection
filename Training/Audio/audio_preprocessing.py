import os
import librosa
import numpy as np
from tqdm import tqdm

# Patch deprecated attributes for compatibility
np.complex = complex
np.float = float

# Parameters
target_sr = 16000
duration = 5
n_mfcc = 13
output_base = "processed_mfcc"
print("Current working directory:", os.getcwd())

def preprocess_and_extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    y_trimmed, _ = librosa.effects.trim(y)
    y_fixed = librosa.util.fix_length(y_trimmed, size=target_sr * duration)
    mfcc = librosa.feature.mfcc(y=y_fixed, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def process_folder(input_folder, label):
    output_folder = os.path.join(output_base, label)
    os.makedirs(output_folder, exist_ok=True)

    audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
    for i, fname in enumerate(tqdm(audio_files, desc=f"Processing {label}")):
        try:
            full_path = os.path.join(input_folder, fname)
            mfcc = preprocess_and_extract_mfcc(full_path)
            np.save(os.path.join(output_folder, f"{label}_{i}.npy"), mfcc)
        except Exception as e:
            print(f"Error processing {fname}: {e}")

# Change this to match your directory
base_dir = r"D:\audio processing\archive\Dataset"
process_folder(os.path.join(base_dir, "Real"), "Real")
process_folder(os.path.join(base_dir, "Fake"), "Fake")
