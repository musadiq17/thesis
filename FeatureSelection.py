import csv
import math
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
    SecDict = {}
    for index, sentence in enumerate(csv_reader):
        documents = index+1
        summary = sentence[13]
        description = sentence[14]
        security = sentence[26]
        SecList = []
        SecList.append(security)

        SecDict[index] = SecList

        bug = summary + description
        lowerBug = bug.lower()
        tokenized_stop = lowerBug.split()
        tokenizedWord = [w for w in tokenized_stop if not w in stop_words and not w in punct]
        tokenizedWord = []
        for w in tokenized_stop:
            if w not in stop_words and w not in punct:
                steem = port.stem(w)
                tokenizedWord.append(steem)
        #print(tokenizedWord)
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
        #print(sentence)
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
        tokenized_stop = lowerBug.split()
        tokenizedWord = [w for w in tokenized_stop if not w in stop_words and not w in punct]
        tokenizedWord = []
        for w in tokenized_stop:
            if w not in stop_words and w not in punct:
                steem = port.stem(w)
                tokenizedWord.append(steem)
        #print(tokenizedWord)

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
    detokenizedWord = []
    for index, sentence in enumerate(csv_reader):

        summary = sentence[13]
        description = sentence[14]
        bug = summary + description
        lowerBug = bug.lower()
        tokenized_stop = lowerBug.split()
        tokenizedWord = [w for w in tokenized_stop if not w in stop_words and not w in punct]
        tokenizedWord = []


        for w in tokenized_stop:
            if w not in stop_words and w not in punct:
                steem = port.stem(w)
                tokenizedWord.append(steem)
        detoken = TreebankWordDetokenizer().detokenize(tokenizedWord)

        #print(detoken)
        detokenizedWord.append(detoken)




    for index, voc in enumerate(allDocumentsNoDublicates):
        count = 0
        #print(detokenizedWord)
        #print(index, voc)
        for sentence in detokenizedWord:
            #print(index,sentence)
            if voc in sentence:
                count += 1
                #print(voc)
        dictOfNoOfDocumentsWithTermInside[index] = (voc, count)
    #print(dictOfNoOfDocumentsWithTermInside)




#-------------Calculate IDF--------------
    dicOfIDFNoDublicates = {}
    for i in range(0, len(normalizeTermFrequency)):
        listOfIdfCalcs = []
        for word in normalizeTermFrequency[i]:
            for x in range(0, len(dictOfNoOfDocumentsWithTermInside)):
                if word[0] == dictOfNoOfDocumentsWithTermInside[x][0]:
                    listOfIdfCalcs.append((word[0], math.log(documents/ dictOfNoOfDocumentsWithTermInside[x][1])))
        dicOfIDFNoDublicates[i] = listOfIdfCalcs

    #print(dicOfIDFNoDublicates)

#-----------------calcualte tf-idf----------------

    dictOfTF_IDF = {}

    for i in range(0, len(normalizeTermFrequency)):
        listOfTF_IDF = []
        TFsentence = normalizeTermFrequency[i]
        #print(TFsentence)
        IDFsentence = dicOfIDFNoDublicates[i]
        #print(IDFsentence)
        for x in range(len(IDFsentence)):
            if TFsentence[x][0] == IDFsentence[x][0]:
               listOfTF_IDF.append((TFsentence[x][0], TFsentence[x][1] * IDFsentence[x][1]))
            #print(TFsentence[x][0])
        dictOfTF_IDF[i] = listOfTF_IDF
    #print(dictOfTF_IDF)

#-------------Combine TF-IDF with Security---------------
    TF_IDF_Sec = []
    for i in range(0, documents):
        TF_IDF = dictOfTF_IDF[i]
        #print(TF_IDF)
        SecTag = SecDict[i]
        #print(SecTag)
        #for x in range(len(SecTag)):
        TF_IDF_Sec.append((TF_IDF,SecTag))
    #print(TF_IDF_Sec)

#------------Filter Security bug report's features---------
    listOfSecFeatures = []
    for x in range(0,documents):
        if TF_IDF_Sec[x][1] == ['1']:
            listOfSecFeatures.append((TF_IDF_Sec[x][0]))
    #print(len(listOfSecFeatures))
    #print(listOfSecFeatures)

#------------Filter Non Security bug report's features---------
    listOfNonSecFeatures = []
    for x in range(0,documents):
        if TF_IDF_Sec[x][1] == ['0']:
            listOfNonSecFeatures.append((TF_IDF_Sec[x][0]))
    #print(listOfNonSecFeatures)
    #print(len(listOfNonSecFeatures[0]))

#-----------Calculate Common features------------------------
    listOfCommonFeatures = []
    for w in range(0,len(listOfSecFeatures)):
        for x in range(0,len(listOfSecFeatures[w])):
            for y in range(0,len(listOfNonSecFeatures)):
                for z in range(0,len(listOfNonSecFeatures[y])):
                    if listOfSecFeatures[w][x][0] == listOfNonSecFeatures[y][z][0]:
                        #print('security words')
                        #print(listOfSecFeatures[w][x])
                        #print('Non Security words')
                        #print(listOfNonSecFeatures[y][z])
                        #print(w,x)
                        listOfCommonFeatures.append((listOfSecFeatures[w][x][0]))


























