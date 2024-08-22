import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from pathlib import Path

def build_generator(latent_dim):
    model = Sequential()
    model.add(layers.Dense(128 * 8 * 8, activation="relu", input_dim=latent_dim))
    model.add(layers.Reshape((8, 8, 128)))
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), padding="same"))
    model.add(layers.Activation('tanh'))
    return model

def build_discriminator(img_shape):
    model = Sequential()
    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same", input_shape=img_shape))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))
    return model

def build_gan(generator, discriminator):
    discriminator.compile(loss="binary_crossentropy", optimizer=Adam(0.0002, 0.5), metrics=["accuracy"])
    discriminator.trainable = False
    
    model = Sequential([generator, discriminator])
    model.compile(loss="binary_crossentropy", optimizer=Adam(0.0002, 0.5))
    return model

def train_gan(generator, discriminator, gan, data, epochs, batch_size, latent_dim, output_dir):
    for epoch in range(epochs):
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_imgs = data[idx]
        real_labels = np.ones((batch_size, 1))
        
        latent_noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_imgs = generator.predict(latent_noise)
        fake_labels = np.zeros((batch_size, 1))
        
        d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_y = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_y)
        
        print(f"{epoch}/{epochs} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss}]")
        
        # Save generated images at intervals
        if (epoch + 1) % (epochs // 10) == 0:
            save_generated_images(generator, output_dir, epoch + 1)


def save_generated_images(generator, output_dir, epoch):
    # Add your code to save generated images
    pass

# Main function to run the GAN training
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a GAN on spectrogram data.")
    parser.add_argument('--epochs', type=int, default=10000, help="Number of epochs to train the GAN.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--latent_dim', type=int, default=100, help="Dimensionality of the latent space.")

    args = parser.parse_args()

    # Define the paths
    script_dir = Path(__file__).parent
    data_file = script_dir / 'spectrogram_data.npy'
    output_dir = script_dir / 'Generated'
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the data
    data = np.load(data_file)
    if data.ndim == 3:
        data = np.expand_dims(data, axis=-1)

    # Set up the GAN
    latent_dim = args.latent_dim
    img_shape = (64, 64, 1)
    
    generator = build_generator(latent_dim)
    discriminator = build_discriminator(img_shape)
    gan = build_gan(generator, discriminator)

    # Train the GAN
    train_gan(generator, discriminator, gan, data, epochs=args.epochs, batch_size=args.batch_size, latent_dim=latent_dim, output_dir=output_dir)
