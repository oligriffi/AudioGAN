import subprocess
import os

def run_script(script_name, *args):
    """Run a script with specified arguments."""
    command = ["python3", script_name] + list(args)
    result = subprocess.run(command, capture_output=True, text=True)
    
    print(f"Running: {' '.join(command)}")
    print(f"stdout: {result.stdout}")
    print(f"stderr: {result.stderr}")
    
    if result.returncode != 0:
        raise RuntimeError(f"Error running {script_name}: {result.stderr}")

def main():
    # Define paths and parameters
    audio_dir = "Input"
    spectrogram_dir = "Spectrograms"
    preprocessed_spectrogram_dir = ""
    generated_spectrogram_dir = "Generated"
    generated_audio_dir = "generatedWav"
    
    # Convert audio to spectrograms
    print("Step 1: Converting audio files to spectrograms...")
    run_script("convert_to_spectrograms.py", audio_dir, spectrogram_dir)
    
    # Preprocess spectrograms
    print("Step 2: Preprocessing spectrograms...")
    run_script("preprocess_spectrograms.py", "--input_folder", spectrogram_dir, "--output_folder", preprocessed_spectrogram_dir)
    
    # Train GAN
    print("Step 3: Training GAN...")
    run_script("train_gan.py", "--data_dir", preprocessed_spectrogram_dir, "--output_dir", generated_spectrogram_dir, "--epochs", "1000", "--batch_size", "32", "--latent_dim", "100")
    
    # Convert generated spectrograms to audio
    print("Step 4: Converting generated spectrograms to audio...")
    run_script("convert_spectrogram_to_wav.py", "--input_folder", generated_spectrogram_dir, "--output_folder", generated_audio_dir)

if __name__ == "__main__":
    main()
