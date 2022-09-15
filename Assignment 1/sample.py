
first_sentence = "Data Science is the sexiest job of the 21st century"
second_sentence = "machine learning is the key for data science"
#split so each word have their own string
first_sentence = first_sentence.split(" ")
second_sentence = second_sentence.split(" ")#join them to remove common duplicate words
total= set(first_sentence).union(set(second_sentence))

wordDictA = dict.fromkeys(set(first_sentence), 0) 
wordDictB = dict.fromkeys(set(second_sentence), 0)
for word in first_sentence:
    wordDictA[word]+=1
    
for word in second_sentence:
    wordDictB[word]+=1

def computeTF(wordDict, doc):
    tfDict = {}
    corpusCount = len(doc)
    for word, count in wordDict.items():
        tfDict[word] = count/float(corpusCount)
    return(tfDict)
#running our sentences through the tf function:
tfFirst = computeTF(wordDictA, first_sentence)
tfSecond = computeTF(wordDictB, second_sentence)
#Converting to dataframe for visualization

def computeIDF(wordDict):
    idfDict = {}
    N = len(wordDict)
    
    for word, val in wordDict.items():
        idfDict[word] = math.log10(N / float(val) + 1)
        
    return(idfDict)
#inputing our sentences in the log file
idfs = computeIDF(wordDictA)
idfsB = computeIDF(wordDictB)
print(idfs, idfsB)