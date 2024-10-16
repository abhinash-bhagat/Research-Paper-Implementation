import tensorflow as tf
from .encoder import EncoderLayer as Encoder, positional_encoding
from .decoder import DecoderLayer as Decoder

# Putting it all together: Transformer
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size, 
                 pe_input, pe_target, dropout_rate=0.1, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                               input_vocab_size, pe_input, dropout_rate)
        
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                               target_vocab_size, pe_target, dropout_rate)
        
        # Final linear layer to generate logits
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
    def call(self, inp, tar, training, 
             enc_padding_mask=None, 
             look_ahead_mask=None, 
             dec_padding_mask=None):
        
        # Encoder output
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        
        # Decoder output
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)  # (batch_size, tar_seq_len, d_model)
        
        # Final linear layer
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        
        return final_output, attention_weights