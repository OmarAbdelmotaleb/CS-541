
string1 = "03/23/18 gta online ban wave : what we know so far : did you were unbanned after reban ?"
string2 = "03/23/18 gta online ban wave : what we know so far : yeah well i got re-banned while reinstalling gta v lol"
string3 = "03/27/2018 - late notes living world—seat of power she 's more lying face-first in tyria , but yeah ."

stringt = "03/23/18 when we play gta online post reban ?"

# for like a trigram, add two pads at the start
# <s> <s> 03/23/18 gta online ban wave : what we know so far : did you were unbanned after reban ? 

strings = [string1, string2, string3]


dict_words = dict()
sentence = "this is a sentence"

n = 3

pad = "<s>"

sentence = "<s> <s> this is a sentence"
words = sentence.split(" ")
print(words[0:n+1])
print(words[0:n])

for i in range(len(words)):
    if n + i > len(words):
        break
    print("i: ", i, " | ", words[i:n+i])
    print("i: ", i, " | ", words[i:n+i-1])


if not dict():
    print("empty")

# if "this is a sentence"
# but n = 5 (5-gram), pad
# "<s> this is a sentence"
# for sentence in strings:
#     words = sentence.split(" ")

#     words_L = len(words)
#     num_pads = n - 1
#     while num_pads > 0:
#         words.insert(0, pad)
#         num_pads -= 1
#         words_L = len(words)

#     for i in range(len(words)):
#         if n+i > words_L:
#             break
#         # Make a string from the list of words
#         # from i to n+i
#         words_s = " ".join(words[i:n+i])
#         dict_words[words_s] += 1

# print(dict_words.items())





# for line in strings:
#     words = line.split(" ")

#     words_L = len(words)

#     # if "this is a sentence"
#     # but n = 5 (5-gram), pad
#     # "<s> this is a sentence"
#     while words_L < n:
#         words.insert(0, pad)
#         words_L = len(words)



# Just do it the hard way. Make a function to break down the sentence into multiple possible phrases
# according to N and add it to a dictionary to record the frequency of it.


# L = string.split(" ")

# N = 6

# # Check first if the length of the list is less than N
# #   if so, add the list to the dict
# # else
# # section off the list starting from 0 to n and use increment counter
# # until n is the length of the list

# dict_words = dict()



# for i in range(len(L)):
#     if N+i > len(L):
#         break
#     words = " ".join(L[i:N+i])
#     if words not in dict_words:
#         dict_words[words] = 1
#     else:
#         dict_words[words] += 1
    
# test = {}
# if not test:
#     print("\aW")


# """

# do you even read what you type before you hit add comment or are you just slow in general ?

# do
# do you
# you
# do you even
# you even
# even



# """

# """
# {
# '03/23/18 gta online ban wave :': 1, 
# 'gta online ban wave : what': 1, 
# 'online ban wave : what we': 1, 
# 'ban wave : what we know': 1, 
# 'wave : what we know so': 1, 
# ': what we know so far': 1, 
# 'what we know so far :': 1, 
# 'we know so far : did': 1, 
# 'know so far : did you': 1, 
# 'so far : did you were': 1, 
# 'far : did you were unbanned': 1, 
# ': did you were unbanned after': 1, 
# 'did you were unbanned after reban': 1, 
# 'you were unbanned after reban ?': 1, 

# 'were unbanned after reban ?': 1, 
# 'unbanned after reban ?': 1, 
# 'after reban ?': 1, 
# 'reban ?': 1, 
# '?': 1}
# """