import tkinter as tk
from tkinter import messagebox, filedialog
import subprocess
import os

# Define the paths to your scripts
SCRIPT_PATHS = {
    "Convert to Spectrograms": "convert_to_spectrograms.py",
    "Preprocess Spectrograms": "preprocess_spectrograms.py",
    "Train GAN": "train_gan.py",
    "Generate Spectrograms": "generate_spectrograms.py",
    "Convert Spectrogram to WAV": "convert_spectrogram_to_wav.py"
}

def run_script(script_name, args=None):
    script_path = SCRIPT_PATHS[script_name]
    command = ["python", script_path]
    
    if args:
        command.extend(args)
    
    try:
        subprocess.run(command, check=True)
        messagebox.showinfo("Success", f"{script_name} ran successfully!")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred while running {script_name}.\n\n{e}")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred.\n\n{e}")

def show_help():
    help_window = tk.Toplevel()
    help_window.title("Help - GAN Parameters and Process")

    help_text = """
    Explanation of GAN Parameters and Process:
    
    1. Epochs:
       - Definition: The number of times the entire training dataset is passed forward and backward through the neural network.
       - Effect: Higher values usually lead to better results but take longer to train.

    2. Batch Size:
       - Definition: The number of training examples utilized in one iteration.
       - Effect: Larger batch sizes make training faster but may require more memory.

    3. Latent Dimension:
       - Definition: The size of the input vector fed into the generator.
       - Effect: A higher latent dimension increases the complexity of the generated outputs.

    Button Explanations and Workflow:
    
    1. Convert to Spectrograms:
       - This button runs the script that converts audio files in the input directory into spectrogram images.
       - The generated spectrograms are saved in the 'Spectograms' folder.

    2. Preprocess Spectrograms:
       - This button processes the spectrogram images to prepare them for training by resizing, normalizing, or applying any other preprocessing steps required by the GAN.
       - The processed spectrograms are saved in a format ready for training.

    3. Train GAN:
       - This button starts the training process for the GAN using the preprocessed spectrograms.
       - The training parameters (Epochs, Batch Size, Latent Dimension) are controlled by the sliders in the main window.
       - The trained GAN model is saved.

    4. Generate Spectrograms:
       - This button generates a specified number of spectrograms using the trained GAN model.
       - The number of spectrograms to generate can be specified by typing a number into the provided field.
       - The generated spectrograms are saved in the 'Generated' folder.

    5. Convert Spectrogram to WAV:
       - This button converts the generated spectrogram images back into audio files (WAV format).
       - The output audio files are saved in the 'generatedWav' folder.
       - The script checks if the WAV file already exists before converting to avoid duplicates.

    Overall Workflow:
       - Start by clicking 'Convert to Spectrograms' to generate spectrograms from your input audio files.
       - Next, click 'Preprocess Spectrograms' to prepare these spectrograms for training.
       - Use the sliders to set your desired GAN training parameters, then click 'Train GAN' to start the training process.
       - After training, click 'Generate Spectrograms' to create new spectrograms using the trained GAN model.
       - Finally, click 'Convert Spectrogram to WAV' to convert the generated spectrograms into audio files.
    """
    
    tk.Label(help_window, text=help_text, justify="left", padx=10, pady=10).pack()

def generate_spectrograms():
    num_spectrograms = int(num_spectrograms_entry.get())
    latent_dim = latent_dim_slider.get()
    output_dir = output_dir_entry.get()
    model_path = "generator_model.h5"

    # Check if the model file exists
    if not os.path.isfile(model_path):
        messagebox.showwarning("Model File Not Found", f"The model file '{model_path}' does not exist. Please ensure the model file is in the correct location.")
        return

    args = [
        "--model_path", model_path,
        "--latent_dim", str(latent_dim),
        "--num_spectrograms", str(num_spectrograms),
        "--output_dir", output_dir
    ]
    run_script("Generate Spectrograms", args)

def create_gui():
    global num_spectrograms_entry, latent_dim_slider, output_dir_entry  # Make these variables global
    
    root = tk.Tk()
    root.title("AudioGAN Interface")

    # GAN parameters
    gan_frame = tk.LabelFrame(root, text="GAN Parameters", padx=10, pady=10)
    gan_frame.pack(padx=10, pady=10, fill="x")

    # Slider for epochs
    tk.Label(gan_frame, text="Epochs:").grid(row=0, column=0, sticky="w")
    epochs_slider = tk.Scale(gan_frame, from_=100, to=10000, resolution=100, orient=tk.HORIZONTAL)
    epochs_slider.set(10000)
    epochs_slider.grid(row=0, column=1)

    # Slider for batch size
    tk.Label(gan_frame, text="Batch Size:").grid(row=1, column=0, sticky="w")
    batch_size_slider = tk.Scale(gan_frame, from_=8, to=128, resolution=8, orient=tk.HORIZONTAL)
    batch_size_slider.set(32)
    batch_size_slider.grid(row=1, column=1)

    # Slider for latent dimension
    tk.Label(gan_frame, text="Latent Dimension:").grid(row=2, column=0, sticky="w")
    latent_dim_slider = tk.Scale(gan_frame, from_=50, to=500, resolution=10, orient=tk.HORIZONTAL)
    latent_dim_slider.set(100)
    latent_dim_slider.grid(row=2, column=1)

    # Input and Output directory selection
    def select_directory(entry):
        directory = filedialog.askdirectory()
        if directory:
            entry.delete(0, tk.END)
            entry.insert(0, directory)

    tk.Label(root, text="Input Directory:").pack(pady=5)
    input_dir_entry = tk.Entry(root, width=50)
    input_dir_entry.pack(pady=5)
    tk.Button(root, text="Browse", command=lambda: select_directory(input_dir_entry)).pack(pady=5)

    tk.Label(root, text="Output Directory:").pack(pady=5)
    output_dir_entry = tk.Entry(root, width=50)
    output_dir_entry.pack(pady=5)
    tk.Button(root, text="Browse", command=lambda: select_directory(output_dir_entry)).pack(pady=5)

    def run_gan():
        epochs = epochs_slider.get()
        batch_size = batch_size_slider.get()
        latent_dim = latent_dim_slider.get()

        args = [
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--latent_dim", str(latent_dim)
        ]
        run_script("Train GAN", args)

    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)

    tk.Button(button_frame, text="Convert to Spectrograms", command=lambda: run_script("Convert to Spectrograms")).pack(side="left", padx=5)
    tk.Button(button_frame, text="Preprocess Spectrograms", command=lambda: run_script("Preprocess Spectrograms")).pack(side="left", padx=5)
    tk.Button(button_frame, text="Train GAN", command=run_gan).pack(side="left", padx=5)
    tk.Button(button_frame, text="Convert Spectrogram to WAV", command=lambda: run_script("Convert Spectrogram to WAV")).pack(side="left", padx=5)

    # New entry for the number of spectrograms to generate
    tk.Label(root, text="Number of Spectrograms to Generate:").pack(pady=5)
    num_spectrograms_entry = tk.Entry(root)
    num_spectrograms_entry.pack(pady=5)
    tk.Button(root, text="Generate Spectrograms", command=generate_spectrograms).pack(pady=10)

    # Add Help button
    tk.Button(root, text="Help", command=show_help).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
