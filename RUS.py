import csv
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from collections import Counter
from FeatureSelection import EndFeatureDict
punct = set(punctuation)
stop_words = set(stopwords.words("english"))
port = PorterStemmer()
detoken = ''



with open('testing.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)

# ---calculate the TF--
# first: tokenize the words

    allTokenizedWord = {}
    SecDict = {}
    Issue_Dict = {}

    for index, sentence in enumerate(csv_reader):
        documents = index+1
        summary = sentence[13]
        description = sentence[14]
        security = sentence[26]
        issue_id = sentence[0]
        issue_List = []
        issue_List.append(issue_id)
        Issue_Dict[index] = issue_List

        SecList = []
        SecList.append(security)

        SecDict[index] = SecList

        bug = summary + description
        lowerBug = bug.lower()

        tokenized_stop = word_tokenize(lowerBug)

        tokenizedWord = [w for w in tokenized_stop if not w in stop_words and not w in punct]
        tokenizedWord = []
        for w in tokenized_stop:
            if w not in stop_words and w not in punct:
                steem = port.stem(w)
                tokenizedWord.append(steem)
        allTokenizedWord[index] = tokenizedWord
    #print(allTokenizedWord)
    NoDublicateDict = {}
    #print(allTokenizedWord)
    for i in range(0, documents):
        NoDubliacteTerms = []
        for word in allTokenizedWord[i]:

            if word not in NoDubliacteTerms:
                NoDubliacteTerms.append(word)
            NoDublicateDict[i] = NoDubliacteTerms
    #print(NoDublicateDict)


#------------Combine issued id, text and security---------------
    Id_Text_Sec = []
    for i in range(0, documents):
        Issue = Issue_Dict[i]
        Text = NoDublicateDict[i]
        Sec = SecDict[i]
        Id_Text_Sec.append((Issue,Text,Sec))

# -----------Write CSV File---------------------------
with open('Features.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)

    Label = ['label']
    Id_No = ['id']
    EndFeatureDictt = Id_No + EndFeatureDict + Label
    fieldnames = EndFeatureDictt
    thewriter = csv.DictWriter(f, fieldnames=fieldnames)
    thewriter.writeheader()


#-----------Check features in bug reports-------------
    totalDict = {}
    for x in range(0,documents):
        #print(Id_Text_Sec[x][0])
        totalList = []

        for y in range(0,len(Id_Text_Sec[x][1])):
            for i in range(0, 100):
                if EndFeatureDict[i] == Id_Text_Sec[x][1][y]:
                    #print(EndFeatureDict[i])
                    val = "1"
                    #print(val)
                    #print(Id_Text_Sec[x][1][y])
                    totalList.append((EndFeatureDict[i]))



        totalDict[x] = totalList

    #print(totalDict)
    #print(EndFeatureDict)
    FeaturesDict = {}
    for x in range(0,100):
        FeaturesDict[x] = EndFeatureDict[x]



#---------------------Testing--------------------------
    print(totalDict)
    print(FeaturesDict)




#---------------------put values in csv file-------------------------
    newList = []
    #print(Id_Text_Sec[1][0][0])
    for i in range(0, documents):
        list = totalDict[i]
        newList.append(list)
    #print(newList[1][2])
    for x in range(0,documents):
        KeyDict = {}
        combineDict = {}
        for y in range(0,len(newList[x])):

            KeyDict[newList[x][y]] = 1
        #print(KeyDict)

        Id_no = {'id': Id_Text_Sec[x][0][0]}
        secLabel = {'label': Id_Text_Sec[x][2][0]}
        combineDict = {**Id_no,**KeyDict, ** secLabel}

        thewriter.writerow(combineDict)














