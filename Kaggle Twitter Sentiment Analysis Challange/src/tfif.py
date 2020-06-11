#!/usr/bin/env python
# coding: utf-8


#import numpy as np 
import pandas as pd 
import re  
import nltk  
nltk.download('stopwords')  
from nltk.corpus import stopwords
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def main():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    #ss = pd.read_csv('sample_submission.csv')
    sns.countplot(x='sentiment', data=train)
    
    X = train.iloc[:, 2].values 
    y = train.iloc[:, 3].values 
    X_test = test.iloc[:, 1].values 
    y_test = test.iloc[:, 2].values 
    
    
    text_array = []
     
    for sententence in range(0, len(X)):  
        # Remove all the special characters
        text = re.sub(r'\W', ' ', str(X[sententence]))
     
        # remove all single characters
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
     
        # Remove single characters from the start
        text = re.sub(r'\^[a-zA-Z]\s+', ' ', text) 
     
        # Substituting multiple spaces with single space
        text= re.sub(r'\s+', ' ', text, flags=re.I)
     
        # Removing prefixed 'b'
        text = re.sub(r'^b\s+', '', text)
     
        # Converting to Lowercase
        text = text.lower()
     
        text_array.append(text)
    
    test_array = []
     
    for test_sent in range(0, len(X_test)):  
        # Remove all the special characters
        test_text = re.sub(r'\W', ' ', str(X[test_sent]))
     
        # remove all single characters
        test_text = re.sub(r'\s+[a-zA-Z]\s+', ' ', test_text)
     
        # Remove single characters from the start
        test_text = re.sub(r'\^[a-zA-Z]\s+', ' ', test_text) 
     
        # Substituting multiple spaces with single space
        test_text= re.sub(r'\s+', ' ', test_text, flags=re.I)
     
        # Removing prefixed 'b'
        test_text = re.sub(r'^b\s+', '', test_text)
     
        # Converting to Lowercase
        test_text = test_text.lower()
     
        test_array.append(test_text)
    
    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  
    X = tfidfconverter.fit_transform(text_array).toarray()
    
    testconverter = TfidfVectorizer(max_features=2000, stop_words=stopwords.words('english'))  
    X_test = testconverter.fit_transform(test_array).toarray()
    
    text_classifier = RandomForestClassifier(n_estimators=30, random_state=1)  
    text_classifier.fit(X, y)
    
    predictions = text_classifier.predict(X_test)
    
    print(confusion_matrix(y_test,predictions))  
    print(classification_report(y_test,predictions))  
    print(accuracy_score(y_test, predictions))
    
    XX_train, XX_test, yy_train, yy_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    text_classifier = RandomForestClassifier(n_estimators=100, random_state=0)  
    text_classifier.fit(XX_train, yy_train)
    
    predictions2 = text_classifier.predict(XX_test)
    
    print(confusion_matrix(yy_test,predictions2))  
    print(classification_report(yy_test,predictions2))  
    print(accuracy_score(yy_test, predictions2))
