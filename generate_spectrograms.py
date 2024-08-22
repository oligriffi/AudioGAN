# generate_spectrograms.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse
import os

def generate_spectrograms(model_path, latent_dim, num_spectrograms, output_dir):
    # Load the trained generator model
    generator = load_model(model_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in range(num_spectrograms):
        # Generate random noise as input
        noise = np.random.normal(0, 1, (1, latent_dim))
        
        # Generate a spectrogram
        generated_spectrogram = generator.predict(noise)
        
        # Rescale spectrogram to [0, 255] and convert to uint8
        generated_spectrogram = (0.5 * generated_spectrogram[0] + 0.5) * 255
        generated_spectrogram = generated_spectrogram.astype(np.uint8)
        
        # Save the spectrogram image
        output_path = os.path.join(output_dir, f"generated_spectrogram_{i}.png")
        tf.keras.preprocessing.image.save_img(output_path, generated_spectrogram)
        print(f"Saved {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained generator model")
    parser.add_argument("--latent_dim", type=int, required=True, help="Dimensionality of the latent space")
    parser.add_argument("--num_spectrograms", type=int, required=True, help="Number of spectrograms to generate")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated spectrograms")
    
    args = parser.parse_args()
    
    # Generate spectrograms
    generate_spectrograms(args.model_path, args.latent_dim, args.num_spectrograms, args.output_dir)
