import csv
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from string import punctuation
from nltk.stem import PorterStemmer
from wicketFeatureExtraction import EndFeatureDict
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
with open('wicketFeatures.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)

    Label = ['label']
    #Id_No = ['id']
    EndFeatureDictt = EndFeatureDict + Label
    fieldnames = EndFeatureDictt
    thewriter = csv.DictWriter(f, fieldnames=fieldnames)
    thewriter.writeheader()


#-----------Check features in bug reports-------------
    totalDict = {}
    totalDict2 = {}
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
        #print(totalList)
        totalDict[x] = totalList
    #print(totalDict)
    #print(EndFeatureDict)
    FeaturesDict = {}
    for x in range(0,100):
        FeaturesDict[x] = EndFeatureDict[x]
    for totalDict_key, totalDict_value in totalDict.items():
        totalList2 = []
        for key, value in FeaturesDict.items():
            if value not in totalDict_value:
                totalList2.append(value)
        #print(totalList2)
        #print(len(totalList2))

        totalDict2[totalDict_key] = totalList2
    #print(totalDict2)


#---------------------put values in csv file-------------------------
    newList = []
    #print(Id_Text_Sec[1][0][0])
    for i in range(0, documents):
        list = totalDict[i]
        newList.append(list)
    newList2 = []
    for i in range(0,documents):
        list2 = totalDict2[i]
        newList2.append(list2)

    #print(newList[1][2])
    for x in range(0,documents):
        KeyDict = {}
        KeyDict2 = {}
        TotalKeyDict = {}
        combineDict = {}
        for y in range(0,len(newList[x])):

            KeyDict[newList[x][y]] = 1
        #print(KeyDict)
        for y in range(0,len(newList2[x])):
            KeyDict2[newList2[x][y]] = 0
        #rint(KeyDict2)
        TotalKeyDict = {**KeyDict, **KeyDict2}
        #print(TotalKeyDict)
        #Id_no = {'id': Id_Text_Sec[x][0][0]}
        secLabel = {'label': Id_Text_Sec[x][2][0]}
        combineDict = {**TotalKeyDict, ** secLabel}
        if combineDict['label'] == '1' or combineDict['label'] == '0':
            #print(combineDict)
            thewriter.writerow(combineDict)














