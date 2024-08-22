# AudioGAN Project
## Overview

This project involves creating and training a Generative Adversarial Network (GAN) to generate audio spectrograms and convert them back into audio files. The workflow includes converting audio files to spectrograms, preprocessing those spectrograms, training the GAN, and then converting the generated spectrograms back into audio.

## Project Structure

```bash
    convert_to_spectrograms.py: Converts audio files into spectrogram images.
    preprocess_spectrograms.py: Preprocesses spectrogram images (e.g., resizing or normalizing).
    train_gan.py: Trains a GAN on the preprocessed spectrograms.
    convert_spectrogram_to_wav.py: Converts generated spectrogram images back into audio files.
    run_all_scripts.py: A master script that runs all the above scripts in sequence.
    gui_interface.py: A GUI for easier interaction with the scripts.
```

## Installation

Clone the repository:

```bash

git clone https://github.com/yourusername/AudioGAN.git
cd AudioGAN

```bash

Create and activate a virtual environment (optional but recommended):

```bash

python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

```bash

Install the required packages:

```bash

pip install numpy matplotlib pillow librosa tensorflow

```bash

## Usage
Running Scripts Individually

    Convert audio files to spectrograms:

    ```bash

python3 convert_to_spectrograms.py --audio_dir "Input" --output_dir "Spectograms"

    --audio_dir: Directory containing the audio files.
    --output_dir: Directory to save the spectrogram images.

```bash

Preprocess spectrogram images:

```bash

python3 preprocess_spectrograms.py --input_folder "Spectograms" --output_folder "PreprocessedSpectrograms"

    --input_folder: Directory containing the raw spectrogram images.
    --output_folder: Directory to save the preprocessed spectrogram images.

```bash

Train the GAN:

```bash

python3 train_gan.py --data_dir "PreprocessedSpectrograms" --output_dir "Generated" --epochs 10000 --batch_size 32 --latent_dim 100

    --data_dir: Directory containing the preprocessed spectrogram images.
    --output_dir: Directory to save the generated spectrograms.
    --epochs: Number of epochs to train the GAN.
    --batch_size: Batch size for training.
    --latent_dim: Dimension of the latent vector.

```bash

Convert generated spectrograms to audio files:

```bash

    python3 convert_spectrogram_to_wav.py --input_folder "Generated" --output_folder "generatedWav"

        --input_folder: Directory containing the generated spectrogram images.
        --output_folder: Directory to save the generated audio files.

```bash

## Using the GUI

The gui_interface.py script provides a graphical user interface for interacting with the scripts. You can use it to convert spectrograms, preprocess them, train the GAN, generate new spectrograms, and convert them back to audio files.

To start the GUI:

```bash

python3 gui_interface.py

```bash

The GUI includes the following features:

    Convert to Spectrograms: Converts audio files to spectrograms.
    Preprocess Spectrograms: Preprocesses spectrogram images for training.
    Train GAN: Trains the GAN model.
    Generate Spectrograms: Generates new spectrograms using the trained GAN model.
    Convert Spectrogram to WAV: Converts the generated spectrograms back to audio files.
    Help: Provides information on how to use the GUI and understand the parameters.

## Notes

    Ensure that all input directories exist and contain the necessary files.
    Modify paths and parameters in the scripts as needed based on your setup and requirements.
    The run_all_scripts.py script assumes that all individual scripts are located in the same directory.