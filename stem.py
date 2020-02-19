from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

port = PorterStemmer()
sample = "he eats what he want to eat for the eating expressed"
a = word_tokenize(sample)
for word in a:
    b = port.stem(word)
    print(b)
