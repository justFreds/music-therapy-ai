import keras
from keras import layers, Model
import tensorflow as tf
import numpy as np
# Import SEQUENCE_LENGTH from preprocess
from maestro_preprocess import SEQUENCE_LENGTH

class VAE(Model):
    """Variational Autoencoder for music generation"""
    
    def __init__(self, input_dim, latent_dim=128, intermediate_dim=256, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        
        # Encoder
        self.encoder = self._build_encoder()
        
        # Decoder
        self.decoder = self._build_decoder()
        
        # Loss tracker
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        
    def _build_encoder(self):
        """Build encoder network"""
        encoder_inputs = layers.Input(shape=(None, self.input_dim))
        
        # LSTM layers for sequence encoding
        x = layers.LSTM(self.intermediate_dim, return_sequences=True)(encoder_inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(self.intermediate_dim)(x)
        x = layers.Dropout(0.2)(x)
        
        # Latent space parameters
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        
        # Sampling layer
        z = Sampling()([z_mean, z_log_var])
        
        encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        return encoder
    
    def _build_decoder(self):
        """Build decoder network"""
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        
        # Repeat latent vector for sequence generation
        x = layers.RepeatVector(SEQUENCE_LENGTH)(latent_inputs)
        
        # LSTM layers for sequence decoding
        x = layers.LSTM(self.intermediate_dim, return_sequences=True)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(self.intermediate_dim, return_sequences=True)(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        decoder_outputs = layers.TimeDistributed(
            layers.Dense(self.input_dim, activation="softmax")
        )(x)
        
        decoder = Model(latent_inputs, decoder_outputs, name="decoder")
        return decoder
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            # Forward pass
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # Compute losses
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.categorical_crossentropy(data, reconstruction),
                    axis=1
                )
            )
            
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1
                )
            )
            
            total_loss = reconstruction_loss + kl_loss
        
        # Backward pass
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


class Sampling(layers.Layer):
    """Sampling layer for reparameterization trick"""
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


