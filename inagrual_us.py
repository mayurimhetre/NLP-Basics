#!/bin/python3

import math
import os
import random
import re
import sys
import zipfile
os.environ['NLTK_DATA'] = os.getcwd() + "/nltk_data"
import nltk


#
# Complete the 'accessTextCorpora' function below.
#
# The function accepts following parameters:
#  1. STRING fileid
#  2. STRING word
#
from nltk.corpus import inaugural
def accessTextCorpora(fileid, word):
    # Write your code here
    a = inaugural.words(fileid)
    n_words = len(a)
    u_words = len(set(a))
    wordcoverage = int(n_words/u_words)
    ed_words = [words for words in set(a) if words.endswith('ed')]
    fileid = [words.lower() for words in a]
    textfreq = [words for words in fileid if words.isalpha()]
    a1 = nltk.FreqDist(textfreq)
    wordfreq = a1[word]
    return wordcoverage,ed_words,wordfreq
    

if __name__ == '__main__':
    fileid = input()
    word = input()

    if not os.path.exists(os.getcwd() + "/nltk_data"):
        with zipfile.ZipFile("nltk_data.zip", 'r') as zip_ref:
            zip_ref.extractall(os.getcwd())

    word_coverage, ed_words, word_freq = accessTextCorpora(fileid, word)

    print(word_coverage)
    print(sorted(ed_words))
    print(word_freq)
	
	
	
	
	
#!/bin/python3

import math
import os
import random
import re
import sys
import nltk


#
# Complete the 'createUserTextCorpora' function below.
#
# The function accepts following parameters:
#  1. STRING filecontent1
#  2. STRING filecontent2
#
from nltk.corpus import PlaintextCorpusReader
def createUserTextCorpora(filecontent1, filecontent2):
    # Write your code here
    with open(os.path.join('nltk_data/','content1.txt'),"w") as file1:
        file1.write(filecontent1) 
        file1.close()
    with open(os.path.join('nltk_data/','content2.txt'),"w") as file2:
        file2.write(filecontent2)  
        file2.close()
    text_corpus = PlaintextCorpusReader('nltk_data/','.*')  
    no_of_words_corpus1 = len(text_corpus.words('content1.txt'))
    no_of_words_corpus2 = len(text_corpus.words('content2.txt'))
    no_of_unique_words_corpus1 = len(set(text_corpus.words('content1.txt')))
    no_of_unique_words_corpus2 = len(set(text_corpus.words('content2.txt')))
    return text_corpus,no_of_words_corpus1,no_of_unique_words_corpus1,no_of_words_corpus2,no_of_unique_words_corpus2


if __name__ == '__main__':
    filecontent1 = input()

    filecontent2 = input()

    path = os.path.join(os.getcwd(), "nltk_data")
    os.makedirs(path, exist_ok=True)
    for file in os.listdir(path):
        os.remove(path+"\\"+file)


    text_corpus, no_of_words_corpus1, no_of_unique_words_corpus1, no_of_words_corpus2, no_of_unique_words_corpus2 = createUserTextCorpora(filecontent1, filecontent2)
    expected_corpus_files = ['content1.txt', 'content2.txt']
    if type(text_corpus) == nltk.corpus.reader.plaintext.PlaintextCorpusReader and sorted(list(text_corpus.fileids())) == expected_corpus_files:
        print(no_of_words_corpus1)
        print(no_of_unique_words_corpus1)
        print(no_of_words_corpus2)
        print(no_of_unique_words_corpus2)
		
		
		
#!/bin/python3

import math
import os
import random
import re
import sys
import zipfile
os.environ['NLTK_DATA'] = os.getcwd()+"/nltk_data"
import nltk

#
# Complete the 'calculateCFD' function below.
#
# The function accepts following parameters:
#  1. STRING_ARRAY cfdconditions
#  2. STRING_ARRAY cfdevents
#

def calculateCFD(cfdconditions, cfdevents):
    # Write your code here
    stop_words= nltk.stopwords.words('english')
    at=[i for i in cfdconditions]
    nt = [(genre, word.lower())
          for genre in cfdconditions
          for word in nltk.corpus.brown.words(categories=genre) if word not in stop_words and   word.isalpha()]

    cdv_cfd = nltk.ConditionalFreqDist(nt)
    cdv_cfd.tabulate(conditions=cfdconditions, samples=cfdevents)
    nt1 = [(genre, word.lower())
          for genre in cfdconditions
          for word in nltk.corpus.brown.words(categories=genre) ]
    
    temp =[]
    for we in nt1:
        wd = we[1]
        if wd[-3:] == 'ing' and wd not in stop_words:
            temp.append((we[0] ,'ing'))

        if wd[-2:] == 'ed':
            temp.append((we[0] ,'ed'))
        

    inged_cfd = nltk.ConditionalFreqDist(temp)
    a=['ed','ing']
    inged_cfd.tabulate(conditions=at, samples=a)
    
    
if __name__ == '__main__':
    cfdconditions_count = int(input().strip())

    cfdconditions = []

    for _ in range(cfdconditions_count):
        cfdconditions_item = input()
        cfdconditions.append(cfdconditions_item)

    cfdevents_count = int(input().strip())

    cfdevents = []

    for _ in range(cfdevents_count):
        cfdevents_item = input()
        cfdevents.append(cfdevents_item)

    if not os.path.exists(os.getcwd() + "/nltk_data"):
        with zipfile.ZipFile("nltk_data.zip", 'r') as zip_ref:
            zip_ref.extractall(os.getcwd())

    calculateCFD(cfdconditions, cfdevents)

#!/bin/python3

import math
import os
import random
import re
import sys
import zipfile
os.environ['NLTK_DATA'] = os.getcwd() + "/nltk_data"
import nltk
import urllib
#
# Complete the 'processRawText' function below.
#
# The function accepts STRING textURL as parameter.
#

def processRawText(textURL):
    # Write your code here
    textcontent = urllib.request.urlopen(textURL).read()
    text_content1 = textcontent.decode('unicode_escape')
    tokenizedlcwords = nltk.word_tokenize(text_content1)
    tokenizedlcwords = [words.lower() for words in tokenizedlcwords ]
    noofwords = len(tokenizedlcwords)
    noofunqwords = len(set(tokenizedlcwords))
    wordcov = int(noofwords/noofunqwords)
    textfreq = [words for words in tokenizedlcwords if words.isalpha()]
    a1 = nltk.FreqDist(textfreq)
    maxfreq = a1.most_common(1)[0][0]
    return noofwords,noofunqwords,wordcov,maxfreq


if __name__ == '__main__':
    textURL = input()

    if not os.path.exists(os.getcwd() + "/nltk_data"):
        with zipfile.ZipFile("nltk_data.zip", 'r') as zip_ref:
            zip_ref.extractall(os.getcwd())

    noofwords, noofunqwords, wordcov, maxfreq = processRawText(textURL)
    print(noofwords)
    print(noofunqwords)
    print(wordcov)
    print(maxfreq)




#!/bin/python3

import math
import os
import random
import re
import sys
import zipfile
os.environ['NLTK_DATA'] = os.getcwd() + "/nltk_data"

import nltk


#
# Complete the 'performBigramsAndCollocations' function below.
#
# The function accepts following parameters:
#  1. STRING textcontent
#  2. STRING word
#

def performBigramsAndCollocations(textcontent, word):
    # Write your code here
    pattern = r'\w+'
    toeknizedwords = nltk.regexp_tokenize(textcontent,pattern)
    toeknizedwords = [words.lower() for words in toeknizedwords]
    tokenizedwordsbigrams = nltk.bigrams(toeknizedwords)
    toekenizednonstopwordsbigrams = [words for words in tokenizedwordsbigrams if words not in nltk.corpus.stopwords.words('english')]
    cfd_bigrams = nltk.ConditionalFreqDist(toekenizednonstopwordsbigrams)
    mostfrequentwordafter = cfd_bigrams[word].most_common(3)
    collectionwords = nltk.text.Text(toeknizedwords).collocation_list()
    return mostfrequentwordafter,collectionwords

if __name__ == '__main__':
    textcontent = input()

    word = input()

    if not os.path.exists(os.getcwd() + "/nltk_data"):
        with zipfile.ZipFile("nltk_data.zip", 'r') as zip_ref:
            zip_ref.extractall(os.getcwd())

    mostfrequentwordafter, collocationwords = performBigramsAndCollocations(textcontent, word)
    print(sorted(mostfrequentwordafter, key=lambda element: (element[1], element[0]), reverse=True))
    print(sorted(collocationwords))
	
	
#!/bin/python3

import math
import os
import random
import re
import sys
import zipfile
os.environ['NLTK_DATA'] = os.getcwd()+"/nltk_data"
import nltk

#
# Complete the 'performStemAndLemma' function below.
#
# The function accepts STRING textcontent as parameter.
#

def performStemAndLemma(textcontent):
    # Write your code here
    pattern = r'\w+'
    toeknizedwords = nltk.regexp_tokenize(textcontent,pattern)
    toeknizedwords = [words.lower() for words in set(toeknizedwords)]
    filteredwords = [words for words in toeknizedwords if words not in nltk.corpus.stopwords.words('english')]
    porterstemmedwords = [nltk.PorterStemmer().stem(words) for words in filteredwords]
    lancasterstemmedwords = [nltk.LancasterStemmer().stem(words) for words in filteredwords]
    lemmatizedwords = [nltk.WordNetLemmatizer().lemmatize(words) for words in filteredwords]
    return porterstemmedwords,lancasterstemmedwords,lemmatizedwords
if __name__ == '__main__':
    textcontent = input()

    if not os.path.exists(os.getcwd() + "/nltk_data"):
        with zipfile.ZipFile("nltk_data.zip", 'r') as zip_ref:
            zip_ref.extractall(os.getcwd())

    porterstemmedwords, lancasterstemmedwords, lemmatizedwords = performStemAndLemma(textcontent)

    print(sorted(porterstemmedwords))
    print(sorted(lancasterstemmedwords))
    print(sorted(lemmatizedwords))
