import os
from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def audio_to_spectrogram(audio_path, output_dir):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    
    # Save the output as a PNG image, replacing the audio extension with .png
    output_path = os.path.join(output_dir, os.path.basename(audio_path).replace(os.path.splitext(audio_path)[-1], '.png'))
    plt.savefig(output_path)
    plt.close()

def convert_audios_to_spectrograms(audio_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3'))]
    
    for audio_file in audio_files:
        audio_path = os.path.join(audio_dir, audio_file)
        audio_to_spectrogram(audio_path, output_dir)
        print(f"Converted {audio_file} to spectrogram")

# Example usage
if __name__ == "__main__":
    # Determine the directory where the script is located
    script_dir = Path(__file__).parent

    # Define input and output directories relative to the script's location
    input_dir = script_dir / "Input"
    output_dir = script_dir / "Spectograms"
    
    # Convert audio files to spectrograms
    convert_audios_to_spectrograms(str(input_dir), str(output_dir))
