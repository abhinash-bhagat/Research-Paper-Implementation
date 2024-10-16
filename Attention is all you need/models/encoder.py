import tensorflow as tf
from  .layers  import MultiHeadAttention
import numpy as np

# Positional Encoding
def get_angles(pos, i, d_model):
    """
    Computes the angles for positional encoding.
    pos: position (0,1,2,...)
    i: dimension
    d_model: total dimension
    """
    angle_rates = 1/np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates
    
def positional_encoding(position, d_model):
    """
    Generates positional encoding vectors.
    position: maximum position (e.g., maximum sentence length)
    d_model: dimension of the model
    """
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model
    )
    
    #Apply sine to even indices in the array; cosine to odd indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...] 
    
    return tf.cast(pos_encoding, dtype=tf.float32)
    
# Encoder Layer
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),  # Feed-Forward layer 1
            tf.keras.layers.Dense(d_model)  # Feed-Forward layer 2
        ])
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, training, mask=None):
        # Multi-Head Attention
        attn_output, _ = self.mha(x, x, x, mask)  # Self-attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # Residual connection + Layer Norm
        
        # Feed-Forward Network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # Residual connection + Layer Norm
        
        return out2