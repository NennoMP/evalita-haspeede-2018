from collections import Counter

def average_sentence_length(dataset) -> int:
    n_lists = len(dataset)
    
    cont = 0
    for token_list in dataset:
        cont += len(token_list)
        
    return cont // n_lists

def median_sentence_length(dataset) -> int:
    sorted_lengths = sorted([len(token_list) for token_list in dataset])
    n = len(sorted_lengths)
    
    if n % 2 == 1:
        return sorted_lengths[n // 2]
    
    left = sorted_lengths[(n - 1) // 2]
    return (left + sorted_lengths[n // 2]) // 2
    
    
def mode_sentence_length(dataset) -> int:
    lengths = [len(token_list) for token_list in dataset]
    
    # Use Counter to get the frequency of each length
    count = Counter(lengths)
    
    # Find the most common length(s)
    common = count.most_common()
    
    # Return the most common length (mode)
    return common[0][0]
    
    
def max_sentence_length(dataset) -> int:
    max_len = -1
    for token_list in dataset:
        max_len = max(max_len, len(token_list))
        
    return max_len