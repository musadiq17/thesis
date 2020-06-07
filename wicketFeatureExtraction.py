import csv
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from string import punctuation
from nltk.stem import PorterStemmer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from collections import Counter
punct = set(punctuation)
stop_words = nltk.corpus.stopwords.words('english')
#print(stop_words)
port = PorterStemmer()
detoken = ''
unwantedTerms = ['size(' ,'.java' ,'boolean.false)4','.setpagestateless(' ,'integer-overflow' ,'!continuetooriginaldestination('  ,':run3' ,'.onsign' ,':8080/login4'
                    ,'setresponsepage(getapplication(' ,'&#8212' ,'.urlfor(final' ,'://localhost' ,':if' ,'.setredirect(true' ,'.onprocessevents' ,'.fix' 'iterator(' ,
                    ').gethomepage(' ,'(yeah' ,'idataprovider-overflow' ,'.steps' ,'.wicket.protocol'  ,'.wicket.requestcycle' , '.service(' ,'.apache' ,'com.ibm.ws',
'hashmap$' ,'.http.']
stop_words = stop_words + unwantedTerms
#print(stop_words)
#print(punct)


with open('wicket.csv', 'r') as csv_file:
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

        tokenized_stop = word_tokenize(lowerBug)
        #print(tokenized_stop)
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
with open('wicket.csv', 'r') as csv_file:
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
        #print(tokenizedWord)

        detoken = TreebankWordDetokenizer().detokenize(tokenizedWord)
        #print(detoken)

        #print(tokenizedWord)

        allDocuments += detoken + ' '
    #print(allDocuments)

    allDocumentsTokenized = word_tokenize(allDocuments)
    #print(allDocumentsTokenized)
    allDocumentsNoDublicates = []
    for word in allDocumentsTokenized:
        if word not in allDocumentsNoDublicates:
            allDocumentsNoDublicates.append(word)
    #print(allDocumentsNoDublicates)

#second: calculate the number of documents where term t appear
dictOfNoOfDocumentsWithTermInside = {}
with open('wicket.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    detokenizedWord = []
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




#-------------Calculate IDF--------------
    dicOfIDFNoDublicates = {}
    for i in range(0, len(normalizeTermFrequency)):
        listOfIdfCalcs = []
        for word in normalizeTermFrequency[i]:
            for x in range(0, len(dictOfNoOfDocumentsWithTermInside)):
                if word[0] == dictOfNoOfDocumentsWithTermInside[x][0]:
                    if dictOfNoOfDocumentsWithTermInside[x][1] != 0:
                        listOfIdfCalcs.append((word[0], math.log(documents/dictOfNoOfDocumentsWithTermInside[x][1])))
        #print(listOfIdfCalcs)
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
    #print(TF_IDF_Sec[0][0])

    # ------------Filter Security bug report's features---------
    listOfSecFeatures = []
    secResult = []
    finalSecList = []
    for x in range(0, documents):
        listOfSecFeatures.append((TF_IDF_Sec[x][0]))
    # print(listOfSecFeatures)
    for x in range(0, len(listOfSecFeatures)):
        for y in range(0, len(listOfSecFeatures[x])):
            secResult.append(listOfSecFeatures[x][y])

    sortedlistSec = sorted(secResult, key=lambda x: x[1])
    sortedlistSec.reverse()
    finalSecList.append(sortedlistSec[:100])
    # print(finalSecList)

    FinalSecList = []
    for x in range(0, 100):
        # print(finalSecList[0][x][0])
        FinalSecList.append(finalSecList[0][x][0])
    # print(FinalSecList)

    EndFeatureList = []

    EndFeatureList = FinalSecList
    # print(EndFeatureList)
    EndFeatureDict = {}
    # print(EndFeatureList)

    EndFeatureDict = EndFeatureList
    print(EndFeatureDict)

















































