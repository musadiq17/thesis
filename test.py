import csv
import itertools
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer
from nltk.tokenize.treebank import TreebankWordDetokenizer
punct = set(punctuation)
stop_words = set(stopwords.words("english"))
port = PorterStemmer()


with open('test.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)

    for index, sentence in enumerate(csv_reader):

        summary = sentence[13]
        description = sentence[14]
        bug = summary + description
        lowerBug = bug.lower()
        tokenized_stop = word_tokenize(lowerBug)
        tokenizedWord = [w for w in tokenized_stop if not w in stop_words and not w in punct]
        tokenizedWord = []
        detokenizedWord = []
        for w in tokenized_stop:
            if w not in stop_words and w not in punct:
                steem = port.stem(w)
                tokenizedWord.append(steem)
        detoken = TreebankWordDetokenizer().detokenize(tokenizedWord)

        #print(detoken)
        detokenizedWord.append(detoken)

        print(detokenizedWord)


