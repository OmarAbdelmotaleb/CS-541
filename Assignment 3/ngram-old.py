#####################################################################
# Omar Abdelmotaleb
# I pledge my honor that I have abided by the Stevens Honor System.
# Uniform Probability with Relative Frequencies
#####################################################################


import pandas as pd
import numpy as np

# dict_words = dict()

RUN_PPL = True
# N between 1 and 7
N = 1

pad = "<s>"

# Add empty token before hand

# Create a dictionary with the keys being the words
# and the values being the frequency of the words

# Read in all of the words from ted.txt into dict_words
# _ _ _ this is a sentence
# if sentence < n make sentence N length
# N=7
def fill_dict(text_file, n):
    dict_words = dict()
    with open(text_file, encoding="utf8") as f:

        for line in f.read().splitlines():

            words = line.split(" ")

            # If the length of the sentence is less than n, just add the sentence
            if len(words) <= n:
                words_s = " ".join(words)
                if words_s not in dict_words:
                    dict_words[words_s] = 1
                else:
                    dict_words[words_s] += 1

            else:
                for i in range(len(words)):
                    if n+i > len(words):
                        break
                    # Make a string from the list of words
                    # from i to n+i
                    words_s = " ".join(words[i:n+i])
                    # Up the frequency in the dictionary
                    if words_s not in dict_words:
                        dict_words[words_s] = 1
                    else:
                        dict_words[words_s] += 1

        f.close()

    return dict_words

def P_word(words, dict_words):
    if words not in dict_words:
        return 1 # Return 1 / N for part 4
        # Add 1                 add N for part 4 laplace
    return dict_words[words] / sum(dict_words.values())

# Where sentence = "this is a sentence"
def P_sentence(sentence, dict_words, hist_words, n):
    probs = list()
    if n <= 1:
        # Map the probabilities regarding dict_words to each word
        for word in sentence:
            probs.append(P_word(word,dict_words))
        
    elif n <= 7:
        # words = ["this", "is", "a", "sentence"]
        words = sentence
        words_s = ""
        if len(words) <= n:
            words_s = " ".join(words)
            top = 1
            bottom = 1
            if words_s in dict_words:
                top = dict_words[words_s]
            if words_s in hist_words:
                bottom = hist_words[words_s]
                
            probs.append(top / bottom)
        else:
            for i in range(len(words)):
                if n+i > len(words):
                    break
                words_s = " ".join(words[i:n+i])
                top = 1
                bottom = 1
                if words_s in dict_words:
                    top = dict_words[words_s]
                if words_s in hist_words:
                    bottom = hist_words[words_s[::-1]]
                
                probs.append(top / bottom)
    else:
        return 0

    # Return the product of all of the values
    return np.product(probs)

# Calculating Perplexity

def perplexity(text_file, dict_file, n):
    sequence = list()
    dict_words = dict()
    hist_words = dict()


    dict_words = fill_dict(dict_file, n)
    if n - 1 > 0:
        hist_words = fill_dict(dict_file, n-1)
    # Create dictionary for history
    # Create dictionary word including history

    with open(text_file, encoding="utf8") as f:
        for line in f.read().splitlines():
            list_test = list()
            for word in line.split(" "):
                list_test.append(word) # From W1, ..., Wn
                
            # Perform for each sentence then average it out
            # Following PPL formula
            # N = len(list_test)
            seq = P_sentence(list_test, dict_words, hist_words, n)
            sequence.append(seq ** (-1/len(list_test)))
        f.close()
    return sum(sequence) / len(sequence)

ted      = "ted.txt"
reddit   = "reddit.txt"

ted_t    = "test.ted.txt"
reddit_t = "test.reddit.txt"
news_t   = "test.news.txt"

if RUN_PPL:

    print("Using ted.txt")
    print(reddit_t + ": ",  perplexity(reddit_t, ted, N))
    print(ted_t    + ": ",  perplexity(ted_t,    ted, N))
    print(news_t   + ": ",  perplexity(news_t,   ted, N))
    print("Using reddit.txt")
    print(reddit_t + ": ",  perplexity(reddit_t, reddit, N))
    print(ted_t    + ": ",  perplexity(ted_t,    reddit, N))
    print(news_t   + ": ",  perplexity(news_t,   reddit, N))