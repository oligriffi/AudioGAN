import tkinter as tk
from tkinter import messagebox
import subprocess
import os

# Define the paths to your scripts
SCRIPT_PATHS = {
    "Convert to Spectrograms": "convert_to_spectrograms.py",
    "Preprocess Spectrograms": "preprocess_spectrograms.py",
    "Train GAN": "train_gan.py",
    "Convert Spectrogram to WAV": "convert_spectrogram_to_wav.py"
}

def run_script(script_name):
    script_path = SCRIPT_PATHS[script_name]
    try:
        # Run the script using subprocess
        subprocess.run(["python3", script_path], check=True)
        messagebox.showinfo("Success", f"{script_name} ran successfully!")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred while running {script_name}.\n\n{e}")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred.\n\n{e}")

def create_gui():
    # Create the main window
    root = tk.Tk()
    root.title("AudioGAN Interface")

    # Create buttons for each script
    for script_name in SCRIPT_PATHS:
        button = tk.Button(root, text=script_name, command=lambda name=script_name: run_script(name))
        button.pack(pady=10)

    # Start the GUI event loop
    root.mainloop()

if __name__ == "__main__":
    create_gui()
