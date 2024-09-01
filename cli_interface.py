import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import soundfile as sf
import argparse

# Define the Generator and Discriminator classes
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

    # Training loop
    for epoch in range(epochs):
        for filename in os.listdir(data_path):
            file_path = os.path.join(data_path, filename)
            
            if os.path.isfile(file_path) and file_path.lower().endswith('.wav'):
                try:
                    # Load and preprocess the audio file
                    audio, sr = sf.read(file_path)
                    audio_len = len(audio)
                    X = []
                    for i in range(0, audio_len - frame_size - 1, frame_shift):
                        frame = audio[i:i + frame_size]
                        if len(frame) == frame_size:
                            X.append(frame)
                    X = np.array(X)
                    X = torch.tensor(X, dtype=torch.float32).to(device)

                    # Ensure the batch size is a multiple of the length of X
                    if len(X) < batch_size:
                        continue  # Skip this file if it doesn't have enough frames

                    # Create real and fake labels
                    valid = torch.ones(batch_size, 1).to(device)
                    fake = torch.zeros(batch_size, 1).to(device)

                    # Train Discriminator
                    optimizer_D.zero_grad()

                    real_audio = X[:batch_size].unsqueeze(1).to(device)
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

                except Exception as e:
                    print(f"Error loading {filename}: {e}")

        # Save generated audio for inspection
        if (epoch + 1) % 10 == 0:
            save_generated_audio(generator, sr, frame_size, epoch + 1)

    # Save the trained model
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')

def save_generated_audio(generator, sr, frame_size, epoch):
    with torch.no_grad():
        noise = torch.randn(1, frame_size, 1)
        generated_audio = generator(noise).cpu().numpy().flatten()
        save_path = f"generated_audio_epoch_{epoch}.wav"
        sf.write(save_path, generated_audio, sr)
        print(f"Saved generated audio to {save_path}")

def generate_audio(model_path, frame_size, num_samples=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(latent_dim=100, output_dim=frame_size).to(device)
    generator.load_state_dict(torch.load(model_path))
    generator.eval()

    with torch.no_grad():
        for i in range(num_samples):
            noise = torch.randn(1, frame_size, 1).to(device)
            generated_audio = generator(noise).cpu().numpy().flatten()
            save_path = f"generated_audio_sample_{i + 1}.wav"
            sf.write(save_path, generated_audio, 44100)
            print(f"Saved generated audio to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Train or generate audio using a GAN.")
    parser.add_argument('command', choices=['train', 'generate'], help='Command to run')
    parser.add_argument('--data_path', type=str, help='Path to the directory containing audio files for training')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--latent_dim', type=int, default=100, help='Dimensionality of the latent space')
    parser.add_argument('--frame_size', type=int, default=2048, help='Size of each frame for the audio')
    parser.add_argument('--frame_shift', type=int, default=512, help='Shift between frames for the audio')
    parser.add_argument('--model_path', type=str, help='Path to the trained model file')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of audio samples to generate')

    args = parser.parse_args()

    if args.command == 'train':
        if args.data_path is None:
            raise ValueError("The --data_path argument is required for training.")
        train_gan_pytorch(
            data_path=args.data_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            latent_dim=args.latent_dim,
            frame_size=args.frame_size,
            frame_shift=args.frame_shift
        )
    elif args.command == 'generate':
        if args.model_path is None:
            raise ValueError("The --model_path argument is required for generating audio.")
        generate_audio(
            model_path=args.model_path,
            frame_size=args.frame_size,
            num_samples=args.num_samples
        )

if __name__ == "__main__":
    main()
