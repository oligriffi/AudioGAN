import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import os
import ntpath
import glob
import soundfile as sf



def start_training():
    data_path = filedialog.askdirectory(title="Select Data Directory")
    epochs = int(epochs_entry.get())
    batch_size = int(batch_size_entry.get())
    latent_dim = int(latent_dim_entry.get())
    frame_size = int(frame_size_entry.get())
    frame_shift = int(frame_shift_entry.get())

    train_gan_pytorch(data_path, epochs, batch_size, latent_dim, frame_size, frame_shift)

root = tk.Tk()
root.title("Audio GAN with PyTorch")

# Add GUI components for adjustable parameters
tk.Label(root, text="Epochs:").grid(row=0, column=0)
epochs_entry = tk.Entry(root)
epochs_entry.grid(row=0, column=1)
epochs_entry.insert(0, "1000")

tk.Label(root, text="Batch Size:").grid(row=1, column=0)
batch_size_entry = tk.Entry(root)
batch_size_entry.grid(row=1, column=1)
batch_size_entry.insert(0, "32")

tk.Label(root, text="Latent Dim:").grid(row=2, column=0)
latent_dim_entry = tk.Entry(root)
latent_dim_entry.grid(row=2, column=1)
latent_dim_entry.insert(0, "100")

tk.Label(root, text="Frame Size:").grid(row=3, column=0)
frame_size_entry = tk.Entry(root)
frame_size_entry.grid(row=3, column=1)
frame_size_entry.insert(0, "2048")

tk.Label(root, text="Frame Shift:").grid(row=4, column=0)
frame_shift_entry = tk.Entry(root)
frame_shift_entry.grid(row=4, column=1)
frame_shift_entry.insert(0, "512")

train_button = tk.Button(root, text="Start Training", command=start_training)
train_button.grid(row=5, columnspan=2)

root.mainloop()
