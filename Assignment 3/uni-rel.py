#####################################################################
# Omar Abdelmotaleb
# I pledge my honor that I have abided by the Stevens Honor System.
# Uniform Probability
#####################################################################


import pandas as pd
import numpy as np

dict_words = dict()

# Create a dictionary with the keys being the words
# and the values being the frequency of the words

# Read in all of the words from ted.txt into dict_words
with open("ted.txt", encoding="utf8") as f:
    for line in f.read().splitlines():
        for word in line.split(" "):
            if word not in dict_words:
                dict_words[word] = 1
            else:
                dict_words[word] += 1
    f.close()

def P_word(word):
    if word not in dict_words:
        return 1 # Return 1 / N for part 4
        # Add 1                 add N for part 4 laplace number of unique unigrams
    return dict_words[word] / sum(dict_words.values())

# Where sentence = ["this", "is", "a", "sentence"]
def P_sentence(sentence):
    # words = sentence.split(" ")
    # Map the probabilities regarding dict_words to each word
    probs = list(map(P_word, sentence))
    # Return the product of all of the values
    return np.product(probs)

# Calculating Perplexity

def perplexity(text_file):
    sequence = list()
    with open(text_file, encoding="utf8") as f:
        for line in f.read().splitlines():
            list_test = list()
            for word in line.split(" "):
                list_test.append(word)
                
                
                # n = len(words)
                # sequence = P_sentence(words)
            # Perform for each sentence then average it out
            # Following PPL formula
            n = len(list_test)
            seq = P_sentence(list_test)
            sequence.append(seq ** (-1/n))
        f.close()
    return sum(sequence) / len(sequence)
    

print("test.reddit.txt: ", perplexity("test.reddit.txt"))
print("test.ted.txt: ", perplexity("test.ted.txt"))
print("test.news.txt: ", perplexity("test.news.txt"))
            
