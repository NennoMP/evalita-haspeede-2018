import keras.backend as K
from keras.layers import Activation, Dense, Flatten, Lambda, Multiply, Permute, RepeatVector

def attention_mechanism(hidden_states, lstm_size):
    """
    Custom Attention layer implementing the attention mechanism from Zhang Y., Wang and Zhang X. Evalita-2018 paper.
    
    Args:
    - hidden_states: biLSTM layers' output
    - lstm_size: biLSTM layers number of units 
    """
    
    # Computing the attention energy for each word feature in the sequence
    # e_i = tanh(Wh * h_i + b_h)
    attention_probs = Dense(1, activation='tanh')(hidden_states)
    
    # Flatten and apply softmax to compute attention weights from energies.
    attention_probs = Flatten()(attention_probs)
    attention_probs = Activation('softmax')(attention_probs)
    
    # Reshape attention weights to match the shape of hidden_states
    attention_probs = RepeatVector(2*lstm_size)(attention_probs)  # reshape to match hidden states
    attention_probs = Permute([2, 1])(attention_probs)
    
    # Multiply the attention weights with the original hidden_states to get weighted representations
    attended_states = Multiply()([hidden_states, attention_probs]) # weighted word features
    
    # Summing over the sequence length to produce the context vector and return it
    return Lambda(lambda x: K.sum(x, axis=1))(attended_states)