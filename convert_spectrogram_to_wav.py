import numpy as np
import librosa
from scipy.io.wavfile import write
from PIL import Image
import argparse
import os

def load_spectrogram_image(image_path):
    # Load the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    spectrogram = np.array(image)

    # Rescale to [-1, 1] if needed
    spectrogram = (spectrogram / 255.0) * 2 - 1
    
    return spectrogram

def resize_spectrogram(spectrogram, target_width):
    original_height, original_width = spectrogram.shape
    if original_width < target_width:
        # Pad the spectrogram if the width is smaller
        pad_width = target_width - original_width
        spectrogram_resized = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
    elif original_width > target_width:
        # Crop the spectrogram if the width is larger
        spectrogram_resized = spectrogram[:, :target_width]
    else:
        # No resize needed
        spectrogram_resized = spectrogram
    
    return spectrogram_resized

def spectrogram_to_audio(spectrogram, n_fft=2048, hop_length=512):
    # Convert from [-1, 1] back to magnitude
    spectrogram = (spectrogram + 1) / 2.0
    
    # Resize the spectrogram to match n_fft
    target_width = n_fft // 2 + 1
    print(f"Original spectrogram shape: {spectrogram.shape}")
    print(f"Resizing to width: {target_width}")
    spectrogram = resize_spectrogram(spectrogram, target_width)
    
    print(f"Resized spectrogram shape: {spectrogram.shape}")
    
    # Check if the spectrogram needs transposing
    if spectrogram.shape[0] != n_fft // 2 + 1:
        spectrogram = spectrogram.T
    
    # Perform inverse STFT
    audio = librosa.istft(spectrogram, hop_length=hop_length, win_length=n_fft, window='hann')
    
    return audio

def save_audio_file(audio, file_path, sample_rate=22050):
    # Save audio as WAV file
    write(file_path, sample_rate, (audio * 32767).astype(np.int16))  # Convert to 16-bit PCM

def process_and_save_audio(image_path, output_path):
    # Load the spectrogram image
    spectrogram = load_spectrogram_image(image_path)
    
    # Convert spectrogram to audio
    audio = spectrogram_to_audio(spectrogram.squeeze())
    
    # Save the audio file
    save_audio_file(audio, output_path)

def convert_all_spectrograms(input_folder, output_folder):
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each .png file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + ".wav"
            output_path = os.path.join(output_folder, output_filename)
            
            print(f"Processing {image_path}...")
            process_and_save_audio(image_path, output_path)
            print(f"Saved {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert spectrogram images to WAV files.")
    parser.add_argument('--input_folder', type=str, required=True, help="Folder containing the spectrogram images.")
    parser.add_argument('--output_folder', type=str, required=True, help="Folder to save the output WAV files.")
    
    args = parser.parse_args()
    
    convert_all_spectrograms(args.input_folder, args.output_folder)
