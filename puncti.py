from string import punctuation
from nltk.tokenize import word_tokenize

punct = set(punctuation)

sample_sentance = "For running a server (daemon)  the current states are: START and STARTED. It would be nice to have transition state: STARTING  and STOPPING."
a = word_tokenize(sample_sentance)
for word in a:
    if word not in punct:
        print(word)


