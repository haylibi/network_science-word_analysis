import english_words    # pip install english-words
import numpy as np

def get_text(
    size: int=10_000
    ,distribution: dict=None    # Dictionary describing what is the probability any given word is chosen    
    ,words: list=list(english_words.get_english_words_set(['web2', 'gcide'], lower=True))
    ,power_law_exponent: float=None      # Power law to use in case distribution has not been given
):
    
    # Generate distribution using power_law if exists one and distribution is None
    if power_law_exponent is not None and distribution is None:
        distribution = [1/(i**power_law_exponent) for i in range(1, len(words)+1)]
        # Update frequencies to match 100% when adding everything up
        total_vals = sum(distribution)
        distribution = [i/total_vals for i in distribution]
        # Some rounding errors may occur, here we mitigate that
        distribution[-1] = 1-sum(distribution[:-1])
    
    # If there is no distribution, assume equal distribution for every single word
    if distribution is None:
        distribution = [1/len(words) for _ in words]

    # If the distribution is dict-like, then this will re-generate the words list 
    # With the correct distribution order (we create a new word list which will have the most
    # probable word of being selected first and, by the end, if there're words left, attribute
    # same probability for every word)
    elif isinstance(distribution, dict):
        d_aux = []
        words_2 = []
        # From dictionary to LIST (distribution)
        for key in distribution.keys():
            d_aux.append(distribution[key]) # Add distribution to list
            words_2.append(key) # Add iterated word to word_list
            words.remove(key)   # Remove said word from original list
        
        # Remainder words will have same probability of being chosen
        for _ in words:
            d_aux.append(1/len(words))
        # Words list should be the ordered words + "leftovers"
        words = words_2 + words
        distribution = d_aux
    

    return ' '.join(np.random.choice(words, replace=True, size=size, p=distribution).tolist())