from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

def plot_sentence_lengths_distribution(dataset_dev, dataset_test, dataset, color_dev='#2986cc', color_test='#f2a900'):
    # compute lengths of the token lists
    dev_lengths = dataset_dev['tokens'].apply(len)
    test_lengths = dataset_test['tokens'].apply(len)

    # Set bin edges: from 0 to max_length with step 1
    bins = np.arange(0, max(max(dev_lengths), max(test_lengths)) + 1, 1)

    # Set up the figure and axis
    plt.figure(figsize=(10, 6))
    plt.hist([dev_lengths, test_lengths], bins, label=['dev', 'test'], align='mid', rwidth=0.8, color=[color_dev, color_test], zorder=2)

    plt.title(f'Sentence length distribution {dataset} dev dataset')
    plt.xlabel('Sentence length')
    plt.ylabel('Number of sentences')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    #plt.savefig(f'{dataset}_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def average_sentence_length(dataset) -> int:
    n_lists = len(dataset)
    
    cont = 0
    for token_list in dataset:
        cont += len(token_list)
        
    return cont // n_lists

def max_sentence_length(dataset) -> int:
    max_len = -1
    for token_list in dataset:
        max_len = max(max_len, len(token_list))
        
    return max_len

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
