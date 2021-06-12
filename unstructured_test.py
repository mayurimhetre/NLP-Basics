#!/usr/bin/env python
# coding: utf-8

# **Run the Cell to import the packages**

# In[1]:


import pandas as pd
import numpy as np
import csv


# **Fill in the Command to load your CSV dataset "imdb.csv" with pandas**

# In[2]:


#Data Loading
imdb=pd.read_csv( 'imdb.csv')
imdb.columns = ["index","text","label"]
print(imdb.head(5))


# **Data Analysis**
# 
# - Get the shape of the dataset and print it.
# 
# - Get the column names in list and print it.
# 
# - Group the dataset by **label** and describe the dataset to understand the basic statistics of the dataset.
# 
# - Print the first three rows of the dataset

# In[5]:


data_size = imdb.shape

print(data_size)

imdb_col_names = imdb.columns

print(imdb_col_names)
print( imdb.groupby('label').describe()        )
print(imdb.head(3)         )


# **Target Identification**
# 
# Execute the below cell to identify the target variables. If 0 it is a bad review,if it is 1 it is a good review.
# 

# In[6]:


imdb_target=imdb['label'] 

print(imdb_target)


# **Tokenization**
# 
# - Convert the text into lower.
# - Tokenize the text using word_tokenize
# - Apply the function **split_tokens** for the column **text** in the **imdb** dataset with axis =1

# In[10]:


from nltk.tokenize import word_tokenize
import nltk
nltk.download('all')


def split_tokens(text):

  message = text.lower()



  
    
  word_tokens = word_tokenize(message)

  return word_tokens

imdb['tokenized_message'] = imdb.apply(lambda row: split_tokens(row['text']),axis=1)


# **Lemmatization**
# 
# - Apply the function **split_into_lemmas** for the column **tokenized_message** with axis=1
# - Print the 55th row from the column **tokenized_message**.
# - Print the 55th row from the column **lemmatized_message**

# In[12]:


from nltk.stem.wordnet import WordNetLemmatizer

def split_into_lemmas(text):

    lemma = []

    lemmatizer = WordNetLemmatizer()

    for word in text:

        a=lemmatizer.lemmatize(word)

        lemma.append(a)

    return lemma

 

imdb['lemmatized_message'] = imdb.apply(lambda row: split_into_lemmas(row['tokenized_message']),axis=1)



print('Tokenized message:',  imdb['tokenized_message'][55]                      )

print('Lemmatized message:',   imdb['lemmatized_message'][55]                   )


# **Stop Word Removal**
# - Set the stop words language as english in the variable **stop_words**
# - Apply the function **stopword_removal** to the column **lemmatized_message** with axis=1
# - Print the 55th row from the column **preprocessed_message**

# In[13]:


from nltk.corpus import stopwords



def stopword_removal(text):

    stop_words = set(stopwords.words('english'))

    filtered_sentence = []

    filtered_sentence = ' '.join([word for word in text if word not in stop_words])

    return filtered_sentence



imdb['preprocessed_message'] = imdb.apply(lambda row: stopword_removal(row['lemmatized_message']),axis=1)

print('Preprocessed message:',imdb['preprocessed_message'][55])

Training_data=pd.Series(list(imdb['preprocessed_message']))

Training_label=pd.Series(list(imdb['label']))


# **Term Document Matrix**
# 
# - Apply CountVectorizer with following parameters
#   - ngram_range = (1,2)
#   - min_df = (1/len(Training_label))
#   - max_df = 0.7
# - Fit the **tf_vectorizer** with the **Training_data**
# - Transform the **Total_Dictionary_TDM** with the **Training_data** 

# In[14]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

tf_vectorizer = CountVectorizer( ngram_range = (1,2), min_df = (1/len(Training_label)),  max_df = 0.7   )

Total_Dictionary_TDM = tf_vectorizer.fit(Training_data)

message_data_TDM = Total_Dictionary_TDM.transform(Training_data)


# **Term Frequency Inverse Document Frequency (TFIDF)**
# - Apply TfidfVectorizer with following parameters
#   - ngram_range = (1,2)
#   - min_df = (1/len(Training_label))
#   - max_df = 0.7
# - Fit the **tfidf_vectorizer** with the **Training_data**
# - Transform the **Total_Dictionary_TFIDF** with the **Training_data** 

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,2),  min_df = (1/len(Training_label))   ,max_df = 0.7      )

Total_Dictionary_TFIDF = tfidf_vectorizer.fit(Training_label)

message_data_TFIDF = Total_Dictionary_TFIDF.transform(Training_label)


# **Train and Test Data**
# 
# Splitting the data for training and testing(90% train,10% test)
# 
# - Perform train-test split on **message_data_TDM** and **Training_label** with 90% as train data and 10% as test      data.

# In[15]:


from sklearn.model_selection import train_test_split#Splitting the data for training and testing

train_data,test_data, train_label, test_label = train_test_split(message_data_TDM,Training_label,test_size=0.1)


# **Support Vector Machine**
# 
# - Get the shape of the train-data and print the same.
# 
# - Get the shape of the test-data and print the same.
# 
# - Initialize SVM classifier with following parameters
#     - kernel = linear
#     - C= 0.025
#     - random_state=seed
# 
# - Train the model with train_data and train_label
# 
# - Now predict the output with test_data
# 
# - Evaluate the classifier with score from test_data and test_label
# 
# - Print the predicted score

# In[18]:


seed=9
from sklearn.svm import SVC

train_data_shape =train_data.shape

test_data_shape = test_data.shape
print("The shape of train data"    ,train_data_shape        )

print("The shape of test data"      ,test_data_shape      )

classifier = SVC(  kernel = 'linear', C= 0.025,        random_state=seed                        )

classifier = classifier.fit(train_data, train_label)

target = classifier.predict(test_data)

score = classifier.score(test_data, test_label)

print('SVM Classifier : ',score)


with open('output.txt', 'w') as file:
    file.write(str((imdb['tokenized_message'][55],imdb['lemmatized_message'][55])))


# **Stochastic Gradient Descent Classifier**
# 
# - Perform train-test split on **message_data_TDM** and **Training_label** with this time 80% as train data and     20% as test data.
# 
# - Get the shape of the train-data and print the same.
# 
# - Get the shape of the test-data and print the same.
# 
# - Initialize SVM classifier with following parameters
#     - loss = modified_huber
#     - shuffle= True
#     - random_state=seed
# 
# - Train the model with train_data and train_label
# 
# - Now predict the output with test_data
# 
# - Evaluate the classifier with score from test_data and test_label
# 
# - Print the predicted score

# In[21]:


from sklearn.linear_model import SGDClassifier

train_data,test_data, train_label, test_label = train_test_split(message_data_TDM,Training_label,test_size=0.2)

train_data_shape =train_data.shape

test_data_shape = test_data.shape

print("The shape of train data",  train_data_shape       )

print("The shape of test data",      test_data_shape      )

classifier =  SGDClassifier(   loss = 'modified_huber', shuffle= True, random_state=seed                         )

classifier = classifier.fit(train_data, train_label)

target=classifier.predict(test_data)

score =  classifier.score(test_data, test_label)

print('SGD classifier : ',score)

with open('output1.txt', 'w') as file:
    file.write(str((imdb['preprocessed_message'][55])))


# In[ ]:




