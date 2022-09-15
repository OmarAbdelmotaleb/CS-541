#####################################################################
# Omar Abdelmotaleb
# I pledge my honor that I have abided by the Stevens Honor System.
# N-Gram LM
#####################################################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# dict_words = dict()

# N between 1 and 7, for testing
OUTPUT = False
N = 7


pad = "<s>"

def fill_dict(text_file, n, history=False):
    """
    Returns a dictionary of words of length n from text_file
    and their frequencies

    Args:
        text_file (string): name of the file to make dictionary from
        n (int): n-gram LM

    Returns:
        dict: Dictionary of words and their frequencies
    """
    dict_words = dict()
    with open(text_file, encoding="utf8") as f:

        for line in f.read().splitlines():
            words = line.split(" ")

            words_L = len(words)
            num_pads = n - 1

            if history:
                num_pads += 1

            # if "this is a sentence"
            # but n = 5 (5-gram), pad
            # "<s> <s> <s> <s> this is a sentence"
            while num_pads > 0:
                words.insert(0, pad)
                num_pads -= 1
                words_L = len(words)
            
            for i in range(len(words)):
                if n + i > words_L:
                    break
                # Make a string for the list of words
                # From i to n + i
                words_s = " ".join(words[i:n+i])
                if words_s not in dict_words:
                    dict_words[words_s] = 1
                else:
                    dict_words[words_s] += 1
        f.close()
    return dict_words

def P(sentence, n, dict_words, history):
    
    """
    P(this, is, a, sentence, .) =~ Product from i = 1 to n=N Pr(wi | history)

    Args:
        sentence (string): A string of words (w1,...,wn)
        n (int): n-gram LM
        dict_words (dict): Frequency of word occurring
        history (dict): Frequency of history occurring
    """

    probs = list()

    words = sentence.split(" ")
    num_pads = n - 1
    # if "this is a sentence"
    # but n = 5 (5-gram), pad
    # "<s> <s> <s> <s> this is a sentence"
    while num_pads > 0:
        words.insert(0, pad)
        num_pads -= 1
    
    for i in range(len(words)):
        if n + i > len(words):
            break
        # Make a string from the list of words
        # From i to n + i
        # <s> <s> this is a sentence
        # n = 3
        # i:  0  |  ['<s>', '<s>', 'this']
        # i:  0  |  ['<s>', '<s>']
        # i:  1  |  ['<s>', 'this', 'is']
        # i:  1  |  ['<s>', 'this']
        # i:  2  |  ['this', 'is', 'a']
        # i:  2  |  ['this', 'is']
        # i:  3  |  ['is', 'a', 'sentence']
        # i:  3  |  ['is', 'a']
        words_s = " ".join(words[i:n+i])
        words_h = " ".join(words[i:n+i-1])

        # Now get the frequency of words_s (being words with history)
        # over frequency of words_h (history)

        top = 1
        bot = 1

        # If the words is not in the dictionary, just set the Pr to 1
        if words_s not in dict_words:
            probs.append(1)
        # If it is,
        else:
            # then set the numerator to its frequency w/ history
            top = dict_words[words_s]

            # if there is no history (unigram)
            if not history:
                # then set denominator to frequency of all words
                bot = sum(dict_words.values())
            else:
                # else, get the frequency of history
                bot = history[words_h]
            
            # Pr(wi | history) = frequency of word wi occuring with history
            # divided by the total frequency of history
            probs.append(top / bot)        
    
    # Return the product of all probabilities
    return np.product(probs)
            
def perplexity(test_file, dict_file, n):
    """
    Calculate the perplexity of a sequence
    of words w1, w2, ..., wn.

    We obtain a list of sequences from test_file.

    PPL = P(w1,w2,...,wn) ^ (-1/n)
    = nth root of 1/P(w1,w2,...,wn)

    Args:
        test_file (string): test txt file
        dict_file (string): file to train LM
        n (int): n-gram LM
    """
    sequence   = list()
    dict_words = dict()
    hist_words = dict()

    # Create dictionary for history
    # Create dictionary word including history
    dict_words = fill_dict(dict_file, n)
    if n - 1 > 0:
        hist_words = fill_dict(dict_file, n-1, history=True)

    with open(test_file, encoding="utf8") as f:

        for sentence in f.read().splitlines():
            seq = P(sentence, n, dict_words, hist_words)

            words = sentence.split(" ")
            sequence.append(seq ** (-1/len(words)))

        f.close()
    
    return sum(sequence) / len(sequence)

ted      = "ted.txt"
reddit   = "reddit.txt"

ted_t    = "test.ted.txt"
reddit_t = "test.reddit.txt"
news_t   = "test.news.txt"


def P_max(sentence, n, dict_words, history):
    
    """
    P(this, is, a, sentence, .) =~ Product from i = 1 to n=N Pr(wi | history)

    Args:
        sentence (string): A string of words (w1,...,wn)
        n (int): n-gram LM
        dict_words (dict): Frequency of word occurring
        history (dict): Frequency of history occurring
    """

    probs = list()

    max_dict = dict()

    words = sentence.split(" ")
    num_pads = n - 1
    # if "this is a sentence"
    # but n = 5 (5-gram), pad
    # "<s> <s> <s> <s> this is a sentence"
    while num_pads > 0:
        words.insert(0, pad)
        num_pads -= 1
    
    for i in range(len(words)):
        if n + i > len(words):
            break
        # Make a string from the list of words
        # From i to n + i
        # <s> <s> this is a sentence
        # n = 3
        # i:  0  |  ['<s>', '<s>', 'this']
        # i:  0  |  ['<s>', '<s>']
        # i:  1  |  ['<s>', 'this', 'is']
        # i:  1  |  ['<s>', 'this']
        # i:  2  |  ['this', 'is', 'a']
        # i:  2  |  ['this', 'is']
        # i:  3  |  ['is', 'a', 'sentence']
        # i:  3  |  ['is', 'a']
        words_s = " ".join(words[i:n+i])
        words_h = " ".join(words[i:n+i-1])

        # Now get the frequency of words_s (being words with history)
        # over frequency of words_h (history)

        top = 1
        bot = 1

        # If the words is not in the dictionary, just set the Pr to 1
        if words_s not in dict_words:
            probs.append(0)
            max_dict[0] = words_s
        # If it is,
        else:
            # then set the numerator to its frequency w/ history
            top = dict_words[words_s]

            # if there is no history (unigram)
            if not history:
                # then set denominator to frequency of all words
                bot = sum(dict_words.values())
            else:
                # else, get the frequency of history
                bot = history[words_h]
            
            # Pr(wi | history) = frequency of word wi occuring with history
            # divided by the total frequency of history
            probs.append(top / bot)        
            max_dict[top/bot] = words_s
    
    # Return the product of all probabilities
    max_val = max(max_dict.keys())
    return (max_dict[max_val], max_val)

def generate_text(text_file, test_file, n):
    output_name = text_file[:-4] + ".out"

    sequence   = list()
    dict_words = dict()
    hist_words = dict()

    # Create dictionary for history
    # Create dictionary word including history
    dict_words = fill_dict(text_file, n)
    if n - 1 > 0:
        hist_words = fill_dict(text_file, n-1, history=True)

    with open(output_name, "w") as o:
        
        first_word = max(dict_words, key=dict_words.get)

        sentence = [first_word]

        context = (n-1) * ['<s>']

        words = " ".join(context + sentence)

        o.write(str(words) + "\n")

        o.close()

generate_text(ted, ted_t, 3)
if OUTPUT: 
    print("Using ted.txt")
    print(reddit_t + ": ",  perplexity(reddit_t, ted,    N))
    print(ted_t    + ": ",  perplexity(ted_t,    ted,    N))
    print(news_t   + ": ",  perplexity(news_t,   ted,    N))
    print("Using reddit.txt")
    print(reddit_t + ": ",  perplexity(reddit_t, reddit, N))
    print(ted_t    + ": ",  perplexity(ted_t,    reddit, N))
    print(news_t   + ": ",  perplexity(news_t,   reddit, N))