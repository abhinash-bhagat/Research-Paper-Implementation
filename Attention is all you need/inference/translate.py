import tensorflow as tf
from models.transformer import Transformer
from utils.data_preparation import create_padding_mask, create_look_ahead_mask
from utils.training_utils import CustomSchedule
import pickle

def load_trained_model(config, checkpoint_path='./checkpoints/train'):
    # Instantiate the Transformer model
    transformer = Transformer(
        num_layers=config['num_layers'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        dff=config['dff'],
        input_vocab_size=config['input_vocab_size'],
        target_vocab_size=config['target_vocab_size'],
        pe_input=config['pe_input'],
        pe_target=config['pe_target'],
        dropout_rate=config['dropout_rate']
    )
    
    # Define the optimizer (necessary for loading checkpoints)
    learning_rate = CustomSchedule(config['d_model'])
    optimizer = tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )
    
    # Restore the latest checkpoint
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored for inference!')
    
    return transformer

def evaluate(transformer, sentence, tokenizer_en, tokenizer_fr, MAX_LENGTH=40):
    sentence = tf.convert_to_tensor(sentence)
    
    sentence = tokenizer_en.encode(sentence.numpy())
    sentence = [tokenizer_en.vocab_size] + sentence + [tokenizer_en.vocab_size + 1]
    encoder_input = tf.expand_dims(sentence, 0)
    
    decoder_input = [tokenizer_fr.vocab_size]
    output = tf.expand_dims(decoder_input, 0)
    
    for i in range(MAX_LENGTH):
        enc_padding_mask = create_padding_mask(encoder_input)
        dec_padding_mask = create_padding_mask(encoder_input)
        
        look_ahead_mask = create_look_ahead_mask(tf.shape(output)[1])
        dec_target_padding_mask = create_padding_mask(output)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        
        predictions, _ = transformer(
            encoder_input, output,
            False,
            enc_padding_mask,
            combined_mask,
            dec_padding_mask
        )
        
        predictions = predictions[:, -1:, :]
        predicted_id = tf.argmax(predictions, axis=-1)
        
        output = tf.concat([output, predicted_id], axis=-1)
        
        if predicted_id == tokenizer_fr.vocab_size + 1:
            break
    
    translated = output.numpy()[0]
    translated = translated[1:]
    translated_sentence = tokenizer_fr.decode([token for token in translated if token < tokenizer_fr.vocab_size])
    
    return translated_sentence

def translate(sentence, transformer, tokenizer_en, tokenizer_fr):
    translated_sentence = evaluate(transformer, sentence, tokenizer_en, tokenizer_fr)
    print(f"Input: {sentence}")
    print(f"Predicted translation: {translated_sentence}")
