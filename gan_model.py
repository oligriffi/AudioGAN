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

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.LSTM(latent_dim, 256, batch_first=True),
            nn.Dropout(0.3),
            nn.LSTM(256, 256, batch_first=True),
            nn.Dropout(0.3),
            nn.LSTM(256, 256, batch_first=True),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(32 * (input_dim // 2), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x.unsqueeze(1))  # Add channel dimension for Conv1D

#Training

def train_gan_pytorch(data_path, epochs, batch_size, latent_dim, frame_size, frame_shift):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    generator = Generator(latent_dim, frame_size).to(device)
    discriminator = Discriminator(frame_size).to(device)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Loss function
    criterion = nn.BCELoss()

    # Load data
    for epoch in range(epochs):
        for fn in glob.iglob(data_path):
            # Load and preprocess the audio file
            audio, sr = sf.read(fn)
            audio_len = len(audio)
            X = []
            for i in range(0, audio_len - frame_size - 1, frame_shift):
                frame = audio[i:i + frame_size]
                X.append(frame)
            X = np.array(X)
            X = torch.tensor(X, dtype=torch.float32).to(device)

            # Create real and fake labels
            valid = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            optimizer_D.zero_grad()

            real_audio = X.unsqueeze(1).to(device)
            fake_audio = generator(torch.randn(batch_size, latent_dim, 1).to(device))

            real_loss = criterion(discriminator(real_audio), valid)
            fake_loss = criterion(discriminator(fake_audio.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()

            gen_loss = criterion(discriminator(fake_audio), valid)
            gen_loss.backward()
            optimizer_G.step()

        print(f"Epoch {epoch + 1}/{epochs} - D Loss: {d_loss.item():.4f}, G Loss: {gen_loss.item():.4f}")

        # Save generated audio for inspection
        if (epoch + 1) % 10 == 0:
            save_generated_audio(generator, sr, frame_size, epoch + 1)

def generate_audio(model_path, frame_size, num_samples=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(latent_dim=100, output_dim=frame_size).to(device)  # Ensure latent_dim is correct
    generator.load_state_dict(torch.load(model_path))
    generator.eval()

    with torch.no_grad():
        for i in range(num_samples):
            noise = torch.randn(1, 100, 1).to(device)  # Ensure latent_dim matches
            generated_audio = generator(noise).cpu().numpy().flatten()
            save_path = f"generated_audio_sample_{i + 1}.wav"
            sf.write(save_path, generated_audio, 44100)
            print(f"Saved generated audio to {save_path}")



