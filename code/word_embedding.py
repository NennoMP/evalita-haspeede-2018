import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


######################################
# W2V
######################################
def get_key_index_mappings(w2v, OOV_TOKEN='<OOV>'):
    key_to_idx, idx_to_key = {OOV_TOKEN: 0}, {0: OOV_TOKEN}
    
    for key, idx in w2v.key_to_index.items():
        # Offset by 1 since <UNK> is at position 0
        key_to_idx[key], idx_to_key[idx+1] = idx + 1, key
        
    return key_to_idx, idx_to_key


def get_embedding_matrix(w2v, idx_to_key=None, OOV_TOKEN='<OOV>'):
    vocab_size, n_dim = w2v.vectors.shape[0] + 1, w2v.vectors.shape[1]
    
    if idx_to_key is None:
        _, idx_to_key = get_key_index_mappings(w2v, OOV_TOKEN)
    
    matrix = np.zeros(shape=(vocab_size, n_dim))
    for idx in idx_to_key:
        # Ignore placeholder for <OOV> words
        if idx != 0:
            matrix[idx] = w2v.get_vector(idx_to_key[idx])
            
    print(f'Embedding matrix shape: {matrix.shape}')
    print(f'{OOV_TOKEN} placeholder for unknown words')
    
    return matrix, vocab_size


######################################
# POS
######################################
def get_key_index_pos_mappings(X, OOV_TOKEN='<OOV>'):
    key_to_idx, idx_to_key = {OOV_TOKEN: 0}, {0: OOV_TOKEN}
    
    unique_pos = set([pos for words in X for pos in words])
    for idx, key in enumerate(sorted(unique_pos)):
        # Offset by 1 since <UNK> is at position 0
        key_to_idx[key], idx_to_key[idx+1]  = idx + 1, key
        
    return key_to_idx, idx_to_key


def get_pos_matrix(idx_to_key_pos): 
    
    idx_to_onehot_pos = {}
    for idx in idx_to_key_pos.keys():
        onehot_enc = [0 for _ in range(len(idx_to_key_pos.keys()) - 1)]
        
        if idx != 0:
            onehot_enc[idx-1] = 1
        idx_to_onehot_pos[idx]  = onehot_enc
    
    return idx_to_onehot_pos

def pos_to_embedding(pos_data, key_to_index_pos, max_text_len, OOV_TOKEN='<OOV>'):
    vocab = key_to_index_pos
    
    X_pos = pos_data
    X_pos = [[word if word in vocab else OOV_TOKEN for word in sentence] for sentence in X_pos]
    X_pos = [[key_to_index_pos[word] for word in sentence] for sentence in X_pos]
    
    X_pos = pad_sequences(X_pos, maxlen=max_text_len, padding='post', truncating='post')
    X_pos = np.array(X_pos)
    
    return X_pos


######################################
# WORDS EMBEDDING
######################################
def sentence_to_embedding(tokens, matrix, key_to_idx, truncation=None, padding=False):
    embeddings = []
    pad_token = [0] * matrix.shape[1]  # Assuming w2v_matrix's columns represent the embedding dimensions

    for token in tokens:
        idx = key_to_idx.get(token, 0)  # Use index 0 (OOV) if word not in vocab
        embeddings.append(matrix[idx])

    # Truncation:
    if truncation is not None:
        embeddings = embeddings[:truncation]

    # Padding:
    if padding and truncation is not None:
        embeddings += [pad_token] * (truncation - len(embeddings))
        
    return np.array(embeddings)

def data_to_embedding(data, matrix, key_to_idx, truncation=None, padding=False):
    X = [sentence_to_embedding(tokens, matrix, key_to_idx, truncation, padding) for tokens in data]
    return np.array(X)


######################################
# FASTTEXT - WORD EMBEDDINGS
######################################
def get_key_index_mappings_ft(ft):
    key_to_idx, idx_to_key = {}, {}
    
    for idx, key in enumerate(ft.words):
        key_to_idx[key], idx_to_key[idx] = idx, key
        
    return key_to_idx, idx_to_key


def get_embedding_matrix_ft(ft, idx_to_key=None):
    vocab_size, n_dim = len(ft.get_words()), ft.get_dimension()
    
    if idx_to_key is None:
        _, idx_to_key = get_key_index_mappings_ft(ft)
    
    matrix = np.zeros(shape=(vocab_size, n_dim))
    for idx in idx_to_key:
        matrix[idx] = ft.get_word_vector(idx_to_key[idx])
            
    print(f'Embedding matrix shape: {matrix.shape}')
    
    return matrix, vocab_size

##################

def extract_char_ngrams(word, min_n, max_n):
    ngrams = []
    for n in range(min_n, max_n + 1):
        ngrams.extend([word[i:i+n] for i in range(len(word) - n + 1)])
    return ngrams

def sentence_to_embedding_ft(tokens, matrix, key_to_idx, truncation=None, padding=False):
    min_n = 3  # Define the minimum n-gram size
    max_n = 6  # Define the maximum n-gram size
    embeddings = []
    pad_token = [0] * matrix.shape[1]  # Assuming w2v_matrix's columns represent the embedding dimensions

    for token in tokens:
        idx = key_to_idx.get(token)

        if idx is not None:
            # If the token exists in the vocabulary
            embeddings.append(matrix[idx])
        else:
            # If the token is out of vocabulary, extract character n-grams
            char_ngrams = extract_char_ngrams(token, min_n, max_n)
            subword_embeddings = [matrix[key_to_idx[subword]] for subword in char_ngrams if subword in key_to_idx]
            if subword_embeddings:
                avg_embedding = np.mean(subword_embeddings, axis=0)
                embeddings.append(avg_embedding)
            else:
                # If no subwords found, use the pad token
                embeddings.append(pad_token)

    # Truncation:
    if truncation is not None:
        embeddings = embeddings[:truncation]

    # Padding:
    if padding and truncation is not None:
        while len(embeddings) < truncation:
            embeddings.append(pad_token)

    return np.array(embeddings)


def data_to_embedding_ft(data, matrix, key_to_idx, truncation=None, padding=False):
    X = [sentence_to_embedding_ft(tokens, matrix, key_to_idx, truncation, padding) for tokens in data]
    return np.array(X)


