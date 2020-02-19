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
detoken = ''


with open('test.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)

# ---calculate the TF--
# first: tokenize the words

    dictOfwords = {}
    for index, sentence in enumerate(csv_reader):
        documents = index+1
        summary = sentence[13]
        description = sentence[14]
        security = sentence[26]
        bug = summary + description
        lowerBug = bug.lower()
        tokenized_stop = word_tokenize(lowerBug)
        tokenizedWord = [w for w in tokenized_stop if not w in stop_words and not w in punct]
        tokenizedWord = []
        for w in tokenized_stop:
            if w not in stop_words and w not in punct:
                steem = port.stem(w)
                tokenizedWord.append(steem)
        #print(tokenizedWord)
        detoken = TreebankWordDetokenizer().detokenize(tokenizedWord)
        #print(detoken)

        dictOfwords[index] = [(word, tokenizedWord.count(word)) for word in tokenizedWord]
    #print(dictOfwords)

# second: remove duplicates
    termFrequency = {}
    for i in range(0,documents):
        listOfNoDublicates = []
        for wordfreq in dictOfwords[i]:
            if wordfreq not in listOfNoDublicates:
                listOfNoDublicates.append(wordfreq)
            termFrequency[i] = listOfNoDublicates

    #print(termFrequency)
# third: normalized term frequency
    normalizeTermFrequency = {}
    for i in range(0, documents):
        sentence = dictOfwords[i]
        lenOfSentence = len(sentence)
        #print(lenOfSentence)

        listOfNormalized = []
        for wordfreq in termFrequency[i]:
            normalizedFrqeuency = wordfreq[1] / lenOfSentence
            listOfNormalized.append((wordfreq[0], normalizedFrqeuency))
            normalizeTermFrequency[i] = listOfNormalized
    #print(normalizeTermFrequency)


# -------------------------------------------------calculate IDF-------------------------------------------------------------------

# First: put togather all sentance and tokenize them
    allDocuments = ''
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
        for w in tokenized_stop:
            if w not in stop_words and w not in punct:
                steem = port.stem(w)
                tokenizedWord.append(steem)
        detoken = TreebankWordDetokenizer().detokenize(tokenizedWord)
        #print(detoken)

        #print(tokenizedWord)

        allDocuments += detoken + ' '
    #print(allDocuments)
    allDocumentsTokenized = allDocuments.split(' ')
    #print(allDocumentsTokenized)
    allDocumentsNoDublicates = []
    for word in allDocumentsTokenized:
        if word not in allDocumentsNoDublicates:
            allDocumentsNoDublicates.append(word)
    #print(allDocumentsNoDublicates)

#second: calculate the number of documents where term t appear
dictOfNoOfDocumentsWithTermInside = {}
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

        for index, voc in enumerate(allDocumentsNoDublicates):
            count = 0
            #print(index, voc)
            for sentence in detokenizedWord:
                print(index,sentence)
                if voc in sentence:
                    count += 1
                    #print(voc)
            dictOfNoOfDocumentsWithTermInside[index] = (voc, count)
        #print(dictOfNoOfDocumentsWithTermInside)


















