# AudioGAN Project

###  This project involves creating and training a Generative Adversarial Network (GAN) to generate audio spectrograms and convert them back into audio files. The workflow includes converting audio files to spectrograms, preprocessing those spectrograms, training the GAN, and then converting the generated spectrograms back into audio.
Project Structure

convert_to_spectrograms.py: Converts audio files into spectrogram images.
preprocess_spectrograms.py: Preprocesses spectrogram images (e.g., resizing or normalizing).
train_gan.py: Trains a GAN on the preprocessed spectrograms.
convert_spectrogram_to_wav.py: Converts generated spectrogram images back into audio files.
run_all_scripts.py: A master script that runs all the above scripts in sequence.

## Installation

Clone the repository:


    git clone https://github.com/yourusername/AudioGAN.git
    cd AudioGAN

Create and activate a virtual environment (optional but recommended):



    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required packages:

    pip install numpy matplotlib pillow librosa tensorflow

## Usage

### Run scripts individually or run all at once with 'python3 run_all_scripts.py'

#### Individually:

Convert audio files to spectrograms:

    python3 convert_to_spectrograms.py --audio_dir "Input" --output_dir "Spectograms"

        --audio_dir: Directory containing the audio files.
        --output_dir: Directory to save the spectrogram images.

Preprocess spectrogram images:


    python3 preprocess_spectrograms.py --input_folder "Spectograms" --output_folder "PreprocessedSpectrograms"
    
        --input_folder: Directory containing the raw spectrogram images.
        --output_folder: Directory to save the preprocessed spectrogram images.

Train the GAN:

    python3 train_gan.py --data_dir "PreprocessedSpectrograms" --output_dir "Generated" --epochs 10000 --batch_size 32 --latent_dim 100
    
        --data_dir: Directory containing the preprocessed spectrogram images.
        --output_dir: Directory to save the generated spectrograms.
        --epochs: Number of epochs to train the GAN.
        --batch_size: Batch size for training.
        --latent_dim: Dimension of the latent vector.

Convert generated spectrograms to audio files:


    python3 convert_spectrogram_to_wav.py --input_folder "Generated" --output_folder "generatedWav"
    
        --input_folder: Directory containing the generated spectrogram images.
        --output_folder: Directory to save the generated audio files.

Run all scripts in sequence:

    python3 run_all_scripts.py
    
        This master script will:
            Convert audio files to spectrograms.
            Preprocess the spectrograms.
            Train the GAN with the preprocessed spectrograms.
            Convert the generated spectrograms back into audio files.

## Notes

    Ensure that all input directories exist and contain the necessary files.
    Modify paths and parameters in the scripts as needed based on your setup and requirements.
    The run_all_scripts.py script assumes that all individual scripts are located in the same directory.
