# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 08:21:25 2018

@author: ilu-pc
"""

import pandas as pd
import pandas as np
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
train_tweets = pd.read_csv('train_tweets.csv') 

def data_cleaing(train_tweets):
    corpus = []
    for i in range(len(train_tweets)):
        review = re.sub('[^a-zA-Z]', ' ', train_tweets['tweet'][i])
        review = re.sub('user','',review)
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in stop_words]
        review = ' '.join(review)
        corpus.append(review)
    return corpus
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
data_xl = data_cleaing(train_tweets)
X = cv.fit_transform(data_xl).toarray()
y = train_tweets.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
#kf = train_test_split(X, y, n_splits=5, n_repeats=10, random_state=None) 
#for train_index, test_index in kf.split(X):
    #X_train, X_test = X[train_index], X[test_index]
    #y_train, y_test = y[train_index], y[test_index]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 123)

# Fitting Naive Bayes to the Training set
#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()
#from sklearn import linear_model
#classifier = linear_model.LogisticRegression(C=1e5,solver= 'newton-cg',max_iter = 200)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, n_jobs=-1)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#f1 score
from sklearn.metrics import f1_score
f1_score = f1_score(y_test, y_pred)  
# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
#test_tweets.csv 
test_tweets = pd.read_csv('test_tweets.csv') 
test_ids = test_tweets.iloc[:,0] 
X_testdata = cv.fit_transform(data_cleaing(test_tweets)).toarray()
y_testdata = classifier.predict(X_testdata)

#update csv file with new prediction
dataout = pd.DataFrame(index = test_ids , data = y_testdata,columns =['label'])
dataout.to_csv('submission_tweets.csv')