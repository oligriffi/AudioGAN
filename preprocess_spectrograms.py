from pathlib import Path
from PIL import Image
import numpy as np

def load_spectrograms(spectrogram_dir, img_shape=(64, 64)):
    spectrogram_files = [f for f in spectrogram_dir.glob('*.png')]
    data = []
    
    for spectrogram_file in spectrogram_files:
        img = Image.open(spectrogram_file).convert('L')  # Convert to grayscale
        img = img.resize(img_shape)  # Resize to the desired shape
        img = np.array(img) / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=-1)  # Add channel dimension
        data.append(img)
    
    return np.array(data)

# Example usage
if __name__ == "__main__":
    script_dir = Path(__file__).parent  # Get the directory where the script is located
    spectrogram_dir = script_dir / "Spectograms"
    
    data = load_spectrograms(spectrogram_dir, img_shape=(64, 64))
    np.save(script_dir / 'spectrogram_data.npy', data)
    print(f"Processed and saved {len(data)} spectrograms.")
