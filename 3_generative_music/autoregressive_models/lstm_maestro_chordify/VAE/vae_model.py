import keras
from keras import layers, Model
import tensorflow as tf

class VAE(Model):
    """Variational Autoencoder for piano music generation"""
    
    def __init__(self, input_dim, sequence_length, latent_dim=64, intermediate_dim=256, 
                 kl_weight=0.0, **kwargs):
        super(VAE, self).__init__(**kwargs)
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self._kl_weight_value = kl_weight
        self.kl_weight = tf.Variable(kl_weight, trainable=False, dtype=tf.float32, name='kl_weight')
        
        # Build encoder and decoder
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
        # Loss trackers
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        
    def _build_encoder(self):
        """Build encoder network - simplified"""
        encoder_inputs = layers.Input(shape=(self.sequence_length,))
        
        # Embedding
        x = layers.Embedding(input_dim=self.input_dim, output_dim=64)(encoder_inputs)
        
        # LSTM encoding
        x = layers.LSTM(self.intermediate_dim, return_sequences=True)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(self.intermediate_dim)(x)
        x = layers.Dropout(0.2)(x)
        
        # Latent space
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        
        return Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    
    def _build_decoder(self):
        """Build decoder network - simplified"""
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        
        # Expand latent
        x = layers.RepeatVector(self.sequence_length)(latent_inputs)
        
        # LSTM decoding
        x = layers.LSTM(self.intermediate_dim, return_sequences=True)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(self.intermediate_dim, return_sequences=True)(x)
        x = layers.Dropout(0.2)(x)
        
        # Output
        decoder_outputs = layers.TimeDistributed(
            layers.Dense(self.input_dim, activation="softmax")
        )(x)
        
        return Model(latent_inputs, decoder_outputs, name="decoder")
    
    def set_kl_weight(self, weight):
        """Set KL weight for annealing"""
        self.kl_weight.assign(weight)
    
    def get_config(self):
        config = super(VAE, self).get_config()
        config.update({
            'input_dim': self.input_dim,
            'sequence_length': self.sequence_length,
            'latent_dim': self.latent_dim,
            'intermediate_dim': self.intermediate_dim,
            'kl_weight': float(self._kl_weight_value)
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
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
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # Reconstruction loss - with numerical stability
            reconstruction_loss = tf.reduce_mean(
                keras.losses.sparse_categorical_crossentropy(
                    data, 
                    reconstruction + 1e-10  # Add epsilon for stability
                )
            )
            
            # KL divergence - simplified, no free bits
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1
                )
            )
            
            # Clip KL loss to prevent explosion
            kl_loss = tf.clip_by_value(kl_loss, 0.0, 1000.0)
            
            # Total loss
            total_loss = reconstruction_loss + self.kl_weight * kl_loss
        
        # Compute gradients
        grads = tape.gradient(total_loss, self.trainable_weights)
        
        # Apply gradients (clipping is in optimizer)
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
    """Sampling layer"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon