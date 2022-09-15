#####################################################################
# Omar Abdelmotaleb
# I pledge my honor that I have abided by the Stevens Honor System.
# Abusive Swearing Detection
#####################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Part 1: Pre-processing

## Lowercasing

df = pd.read_csv("swad_train.csv", nrows=500)
df_test = pd.read_csv("swad_test.csv")

# Lowercase each tweet
tweets = [tweet.lower() for tweet in df["Tweet"].tolist()]
tweets_test = [tweet.lower() for tweet in df_test["Tweet"].tolist()]
# Assign 0 to No, 1 to Yes
labels = [0 if "N" in l else 1 for l in df["Label"].tolist()] 
labels_test = [0 if "N" in l else 1 for l in df_test["Label"].tolist()] 

## Obtains a list of all punctuation present in punctuation.txt
punctuation = []
with open("punctuations.txt") as f:
    punctuation = f.read().splitlines()
    f.close()

## Adds a space before and after a character. Intended for punctuation.
def space(chr):
    return " " + chr + " "

## Stopwords
stopwords = []
with open("stopwords.txt") as f:
    stopwords = f.read().splitlines()
    f.close()

## Tokenization and Stopword Removal
# for i in range(len(tweets)):
#     tweet = tweets[i]
#     tweet = "".join(list(map(lambda c: space(c) if c in punctuation else c, tweet)))
    
#     for stopword in stopwords:
#         tweet = tweet.replace(space(stopword), " ")

#     tweets_len += len(tweet)
#     tweets[i] = tweet

def pre_process(T): 
    for i in range(len(T)):
        tweet = T[i]
        tweet = "".join(list(map(lambda c: space(c) if c in punctuation else c, tweet)))
        
        for stopword in stopwords:
            tweet = tweet.replace(space(stopword), " ")

        T[i] = tweet
    return T

def get_tweets_length(T):
    length = 0
    for i in range(len(T)):
        length += len(T[i])
    return length

tweets = pre_process(tweets)
tweets_test = pre_process(tweets_test)
tweets_len = get_tweets_length(tweets)
tweets_test_len = get_tweets_length(tweets_test)

# print(tweets)
# print(tweets_test)
# print(tweets_len)
# print(tweets_test_len)

# Create a set from the list of tweets
# set_tweets = set(tweets[0].split(" "))
#     for tweet in tweets:
#         set_tweets = set_tweets.union(set(tweet.split(" ")))
def create_set(T):
    set_tweets = set(T[0].split(" "))
    for tweet in T:
        set_tweets = set_tweets.union(set(tweet.split(" ")))
    return set_tweets


    
# Term Frequency
# wordDict = dict.fromkeys(set_tweets,0)

# for tweet in tweets:
#     for word in tweet.split(" "):
#         if word in wordDict.keys():
#             wordDict[word] += 1

# def create_wordDict(T, set_tweets):
#     wordDict = dict.fromkeys(set_tweets,0)
#     for tweet in T:
#         for word in tweet.split(" "):
#             if word in wordDict.keys():
#                 wordDict[word] += 1
#     return wordDict

# tweets_wordDict = create_wordDict(tweets, tweets_set)
# tweets_test_wordDict = create_wordDict(tweets_test, tweets_test_set)
def create_wordDict(tweet, set_tweets):
    wordDict = dict.fromkeys(set_tweets,0)
    for word in tweet.split(" "):
        if word in wordDict.keys():
            wordDict[word] += 1

    return wordDict

def wordDicts(T, set_tweets):
    Dicts = []
    # tweet_set = create_set(T)
    for tweet in T:
        Dicts.append(create_wordDict(tweet, set_tweets))
    return Dicts
tweet_set = create_set(tweets)
tweets_wordDicts = wordDicts(tweets, tweet_set)
tweets_test_wordDicts = wordDicts(tweets_test, tweet_set)


# Compute tf
# tfDict = {}
# for word, count in wordDict.items():
#     tfDict[word] = count/float(tweets_len)

# def tf(wordDict, length):
#     tfDict = {}
#     for word, count in wordDict.items():
#         tfDict[word] = count/float(length)
#     return tfDict

def computeTF(wordDicts, T):
    tfDicts = []
    for i in range(len(T)):
        tfDict = {}
        corpusCount = len(T[i])
        for word, count in wordDicts[i].items():
            tfDict[word] = count/float(corpusCount)
        tfDicts.append(tfDict)
    return tfDicts

# tweets_tf = tf(tweets_wordDict, tweets_len)
# tweets_test_tf = tf(tweets_test_wordDict, tweets_test_len)

tweets_tf = computeTF(tweets_wordDicts, tweets)
tweets_test_tf = computeTF(tweets_test_wordDicts, tweets_test)



# Compute idf        
# n = len(wordDict)
# idfDict = wordDict.copy()
# for word, val in idfDict.items():
#     idfDict[word] = math.log10(n / (float(val) + 1))

def idf(wordDicts):
    idfDicts = []
    for wordDict in wordDicts:
        n = len(wordDict)
        # idfDict = wordDict.copy()
        idfDict = dict.fromkeys(wordDict.keys(), 0)
        for word, val in idfDict.items():
            idfDict[word] = math.log10(n / (float(val) + 1))
        idfDicts.append(idfDict)
    return idfDicts

tweets_idf = idf(tweets_wordDicts)
tweets_test_idf = idf(tweets_test_wordDicts)

# Compute tfidf

# tfidf = {}
# for word, val in tfDict.items():
#     tfidf[word] = val * idfDict[word]

# def tfidf(tfDict, idfDict):
#     tfidf = {}
#     for word, val in tfDict.items():
#         tfidf[word] = val * idfDict[word]
#     return tfidf
def tfidf(tfDicts, idfDicts):
    tfidfs = []
    for i in range(len(tfDicts)):
        tfidf = {}
        for word, val in tfDicts[i].items():
            tfidf[word] = val * idfDicts[i][word]
        tfidfs.append(tfidf)
    return tfidfs

tweets_tfidf = tfidf(tweets_tf, tweets_idf)
tweets_test_tfidf = tfidf(tweets_test_tf, tweets_test_idf)

tfidf_df = pd.DataFrame(tweets_tfidf)
# tfidf_df.to_csv("tfidf.csv")
tfidf_df = tfidf_df.to_numpy()


tfidf_test_df = pd.DataFrame(tweets_test_tfidf)
# tfidf_test_df.to_csv("tfidf_test.csv")
tfidf_test_df = tfidf_test_df.to_numpy()

# print(tfidf_df.shape)
# print(tfidf_test_df.shape)

## Logistic Regression Model
# Adapted from https://www.youtube.com/watch?v=JDU3AzH3WKg&ab_channel=PythonEngineer
class LogisticRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr # Learning Rate
        self.n_iters = n_iters # Num. of iterations for descent
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        # init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted-y))
            db = (1 / n_samples) * np.sum(y_predicted-y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

def accuracy(y_true, y_pred):
    # accuracy = np.sum(y_true == y_pred) / len(y_true)

    accuracy = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            accuracy += 1
    return accuracy / len(y_true)

# tweets_tfidf_X = np.split(np.array(tweets_tfidf.values()), 3588)

# tweets_test_tfidf_Y = np.split(np.array(tweets_test_tfidf.values()), 2)

# print(thing.shape)

# with open("output.txt", "w") as o:
#     o.write(str(tweets))
#     o.close()

regressor = LogisticRegression(lr=0.1, n_iters = 1000)
regressor.fit(tfidf_df, labels)
predictions = regressor.predict(tfidf_test_df)

print("LR classification accuracy: ", accuracy(labels_test, predictions))