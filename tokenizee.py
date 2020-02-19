from nltk.tokenize import word_tokenize, sent_tokenize
sample_sentance = "For running a server (daemon)  the current states are: START and STARTED. It would be nice to have transition state: STARTING  and STOPPING."
sample_sentance1 = "For running a server (daemon)  the current states are: START and STARTED. It would be nice to have transition state: STARTING  and STOPPING."
a = word_tokenize(sample_sentance)
print(a)
b = sent_tokenize(sample_sentance)
print(b) 