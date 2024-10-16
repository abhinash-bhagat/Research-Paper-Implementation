import tensorflow as tf
import numpy as np

# Scaled Dot Product Attenntion Layer
# Creates a custom layer class that inherits from TensorFlow's base Layer class
class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        # calls the parent class's constructor
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        
    def call(self, Q, K, V, mask=None):
        # Calculate the scaled dot product attention between Queries & Keys
        matmul_qk = tf.matmul(Q, K, transpose_b=True) # Shape: (..., seq_len_q, seq_len_k)
        
        # Scale the dot products
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Apply the mask (if provided)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9) # Large negative values to mask
            
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) # Shape: (..., seq_len_q, seq_len_k)
        
        # Multiply the attention weights with the values
        output = tf.matmul(attention_weights, V) # Shape: (..., seq_len_q, seq_len_v)
        
        return output, attention_weights
    
# Multi HeadAttention Layer
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads # Dimension of each head
        
        # Create multiple dense layer for linear projections
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        # Create scaled dot product attention layer
        self.scaled_dot_product_attention = ScaledDotProductAttention()
        self.dense = tf.keras.layers.Dense(d_model) # Final linear layer
        
    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result to shape (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, Q, K, V, mask=None):
        batch_size = tf.shape(Q)[0]
        
        # Linear projections
        Q = self.wq(Q) # Shape: (batch_size, seq_len, d_model)
        K = self.wk(K) # Shape: (batch_size, seq_len, d_model)
        V = self.wv(V) # Shape: (batch_size, seq_len, d_model)
        
        # Split heads
        Q = self.split_heads(Q, batch_size) # Shape: (batch_size, num_heads, seq_len, depth)
        K = self.split_heads(K, batch_size) # Shape: (batch_size, num_heads, seq_len, depth)
        V = self.split_heads(V, batch_size) # Shape: (batch_size, num_heads, seq_len, depth)
        
        # Apply Scaled Dot-Product Attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3]) # Shape: (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model)) # Shape: (batch_size, seq_len_q, d_model)
        
        # Final Linear projection
        output = self.dense(concat_attention) # Shape: (batch_size, seq_len_q, d_model)
        
        return output, attention_weights