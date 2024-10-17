import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore
import numpy as np

# 1. Model Setup

class LoRALayer(layers.Layer):
    def __init__(self, rank, units, **kwargs):
        super(LoRALayer, self).__init__(**kwargs)
        self.rank = rank
        self.units = units
        
    def build(self,input_shape):
        # The original pre-trained weight (frozen)
        self.W0 = self.add_weight(shape=(input_shape[-1], self.units),
                                     initializer='glorot_uniform',
                                     trainable=False,
                                     name  = "W0")# freezing the pretrained model weight
        
        # LoRA's Low rank matrices (A and B)
        self.A=self.add_weight(
            shape=(input_shape[-1], self.rank),
            initializer='glorot_uniform',
            trainable=True, # This will be fine-tuned
            name  = "A"
        )
        
        self.B=self.add_weight(
            shape=(self.rank,  self.units),
            initializer='glorot_uniform',
            trainable=True, # This will be fine-tuned
            name  = "B"
        )
        
    def call(self, inputs):
        # Original forward pass using W0 (frozen)
        output_W0 = tf.matmul(inputs, self.W0)
        
        # LoRA's update: A * B
        output_lora = tf.matmul(tf.matmul(inputs, self.A), self.B)
        
        # Sum the original output with the LoRA update
        return output_W0 + output_lora
    
    
# 2. Transformer Attention Block with LoRA

class MultiHeadAttentionWithLoRA(layers.Layer):
    def __init__(self, embed_dim, num_heads, rank):
        super(MultiHeadAttentionWithLoRA, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.rank = rank
        
        assert embed_dim % num_heads == 0
        self.projection_dim  = embed_dim // num_heads
        
        self.query_dense = LoRALayer(rank=self.rank, units=self.embed_dim)
        self.key_dense = LoRALayer(rank=self.rank, units=self.embed_dim)
        self.value_dense = LoRALayer(rank=self.rank, units=self.embed_dim)
        self.combine_heads = layers.Dense(self.embed_dim)
        
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        
        return output
    
    def split_heads(self, input, batch_size):
        input = tf.reshape(input, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(input, [0, 2, 1, 3])
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        
        # Apply LoRA transformations for query, key, and value projections
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        # Split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        attention_output = self.attention(query, key, value)
        attention_output = tf.transpose(attention_output, [0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.embed_dim))
        
        output = self.combine_heads(concat_attention)
        
        return output
    
    
#  3. Building the Transformer Block

class TransformerBlockWithLoRA(layers.Layer):
    def __init__(self, embed_dim, num_heads, rank, ff_dim, rate=0.1):
        super(TransformerBlockWithLoRA, self).__init__()
        self.attention = MultiHeadAttentionWithLoRA(embed_dim, num_heads, rank)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        
    def call(self, inputs, training):
        attn_output = self.attention(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    
# 4. Final Model with LoRA Transformer Blocks

def  create_lora_transformer_model(vocab_size, max_len, embed_dim, num_heads, ff_dim, rank, num_blocks):
    inputs = layers.Input(shape=(max_len,))
    
    # Embedding layer
    embedding_layer = layers.Embedding(vocab_size, embed_dim)(inputs)
    
    # Stacked transformer blocks
    x = embedding_layer
    for _ in range(num_blocks):
        x = TransformerBlockWithLoRA(embed_dim, num_heads, ff_dim, rank)(x)
    
    # Global pooling and output layer
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(vocab_size, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

