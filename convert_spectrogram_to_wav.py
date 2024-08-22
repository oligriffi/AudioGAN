import os
from PIL import Image
import numpy as np
import librosa
import soundfile as sf
import argparse

def spectrogram_to_audio(spectrogram, n_fft=2048, hop_length=512):
    if len(spectrogram.shape) == 2:
        spectrogram = np.expand_dims(spectrogram, axis=-1)
    
    audio = librosa.istft(spectrogram.squeeze(), hop_length=hop_length, win_length=n_fft, window='hann')
    return audio

def process_and_save_audio(image_path, output_path):
    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Skipping conversion.")
        return

    try:
        # Load and process the spectrogram image
        print(f"Processing {image_path}")
        spectrogram = Image.open(image_path).convert('L')
        spectrogram = np.array(spectrogram)
        print(f"Spectrogram shape: {spectrogram.shape}")

        # Convert the spectrogram to audio
        audio = spectrogram_to_audio(spectrogram)
        print(f"Audio length: {len(audio)} samples")

        # Save the audio to a WAV file
        sf.write(output_path, audio, 22050)
        print(f"Converted {image_path} to {output_path}")

    except Exception as e:
        print(f"An error occurred while processing {image_path}: {e}")

def convert_all_spectrograms(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files_converted = 0
    print(f"Scanning directory: {input_dir}")
    
    for file_name in os.listdir(input_dir):
        print(f"Found file: {file_name}")  # Debug: print each file found
        if file_name.lower().endswith('.png'):
            image_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name.replace('.png', '.wav'))
            process_and_save_audio(image_path, output_path)
            files_converted += 1

    print(f"Total files converted: {files_converted}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert all PNG spectrograms to WAV files.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing the spectrogram PNG files.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the converted WAV files.")
    
    args = parser.parse_args()

    convert_all_spectrograms(args.input_dir, args.output_dir)
