#from nltk.tokenize import word_tokenize
#from nltk.corpus import stopwords

#stop_words = set(stopwords.words("english"))

#sample_sentance = "For running a server (daemon)  the current states are: START and STARTED. It would be nice to have transition state: STARTING  and STOPPING."
#a = word_tokenize(sample_sentance)
#for word in a:
#    if word not in stop_words:
#        print(word)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sent = "This is a sample sentence, showing off the stop words filtration."

stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example_sent)

filtered_sentence = [w for w in word_tokens if not w in stop_words]
filtered_sentence = []
for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)
#print(word_tokens)
print(filtered_sentence)