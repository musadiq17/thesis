import math
documents = ['the the universe has many very stars',
             'the galaxy contains many stars',
             'the cold breeze of winter made it very cold outside'
            ]
#---calculate the term frequency--
#first: tokenize the words
#print(documents)
dictOfwords = {}

for index,sentence in enumerate(documents):
    tokenizedWord = sentence.split(' ')
    dictOfwords[index] = [(word,tokenizedWord.count(word)) for word in tokenizedWord ]
#print(dictOfwords)
#second: remove dupocates
termFrequency = {}
for i in range(0, len(documents)):
    listOfNoDublicates = []
    for wordfreq in dictOfwords[i]:
        if wordfreq not in listOfNoDublicates:
            listOfNoDublicates.append(wordfreq)
        termFrequency[i] = listOfNoDublicates
#print(termFrequency)

#third: normalized term frequency
normalizeTermFrequency = {}
for i in range(0, len(documents)):
    sentence = dictOfwords[i]
    lenOfSentence = len(sentence)

    listOfNormalized = []
    for wordfreq in termFrequency[i]:
        normalizedFrqeuency = wordfreq[1]/lenOfSentence
        listOfNormalized.append((wordfreq[0],normalizedFrqeuency))
        normalizeTermFrequency[i] = listOfNormalized

#print(normalizeTermFrequency)

#--calculate IDF

#First: put togather all sentance and tokenize them
allDocuments = ''
for sentence in documents:

    allDocuments += sentence + ' '
allDocumentsTokenized = allDocuments.split(' ')
#print(allDocumentsTokenized)

allDocumentsNoDublicates = []
for word in allDocumentsTokenized:
    if word not in allDocumentsNoDublicates:
        allDocumentsNoDublicates.append(word)
#print(allDocumentsNoDublicates)

#second: calculate the number of documents where term t appear
print(documents)
dictOfNoOfDocumentsWithTermInside = {}
for index, voc in enumerate(allDocumentsNoDublicates):
    count = 0
    for sentence in documents:
        #print(index,sentence)
        if voc in sentence:
            count += 1
            #print(voc)
    dictOfNoOfDocumentsWithTermInside[index] = (voc, count)
#print(dictOfNoOfDocumentsWithTermInside)

#calcualte IDF

dicOfIDFNoDublicates = {}
for i in range(0,len(normalizeTermFrequency)):
    listOfIdfCalcs = []
    for word in normalizeTermFrequency[i]:
        for x in range(0, len(dictOfNoOfDocumentsWithTermInside)):
            if word[0] ==  dictOfNoOfDocumentsWithTermInside[x][0]:
                listOfIdfCalcs.append((word[0],math.log(len(documents)/dictOfNoOfDocumentsWithTermInside[x][1])))
    dicOfIDFNoDublicates[i] = listOfIdfCalcs

#print(dicOfIDFNoDublicates)

#calcualte tf-idf

dictOfTF_IDF = {}

for i in range(0, len(normalizeTermFrequency)):
    listOfTF_IDF = []
    TFsentence = normalizeTermFrequency[i]
    IDFsentence = dicOfIDFNoDublicates[i]
    for x in range(len(TFsentence)):
        listOfTF_IDF.append((TFsentence[x][0],TFsentence[x][1]*IDFsentence[x][1]))
    dictOfTF_IDF[i] = listOfTF_IDF
#print(dictOfTF_IDF)



