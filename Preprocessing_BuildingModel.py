#!/usr/bin/env python
# coding: utf-8

import re
import string
from requests import get
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ipykernel import kernelapp as app
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import validation_curve
from sklearn.externals import joblib
from sklearn import metrics
from lime.lime_text import LimeTextExplainer

get_ipython().run_line_magic('matplotlib', 'inline')


df = pd.read_csv('Data/Dataset.csv')
df = pd.concat([df,pd.DataFrame(columns=['title', 'description', 'keywords'])], sort=True)
df = df[['link', 'title', 'description', 'keywords', 'class']]

from requests.exceptions import ConnectionError

for i in range(880, len(df)):
    try:
        response = get(df.link[i], allow_redirects=False)
        if response.status_code < 400:
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')
            if soup.title:
                df['title'][i] = soup.title.string
            else:
                df['title'][i] = ''
            for tag in soup.find_all("meta"):
                if tag.get("name", None) == "keywords":
                    df['keywords'][i] = tag.get("content", None)
                if tag.get("name", None) == "description":
                    df['description'][i] = tag.get("content", None)
    except ConnectionError as e:
        df['title'][i] = "No Response"
        df['keywords'][i] = "No Response"
        df['description'][i] = "No Response"
                
df.to_csv('Data/features.csv', index=False)

# Read features.csv
#df = pd.read_csv('features.csv')
df_processed = df
# comine the 3 columns into one column
df_processed['text'] = df_processed[df_processed.columns[1:4]].apply(lambda x: ','.join(x.dropna().astype(str).astype(str)),axis=1)
df_processed.drop(['title', 'description', 'keywords'], inplace=True, axis=1)
# Remove Punctuation
remove_punctuation = '|'.join([re.escape(x) for x in string.punctuation])
df_processed['text'] = df_processed['text'].str.replace(remove_punctuation,"")
# Lowercase text
df_processed['text'] = df_processed['text'].str.lower()
# Remove numbers
df_processed['text'] = df_processed['text'].str.replace('\d+', '')
# Save Processed features
df_processed.to_csv('Data/processed_features.csv', index=False)


"""
Logistic Regression -> Solver='lbfgs'
"""
df_processed = df_processed.sample(frac=1)
x = df_processed.iloc[:, -1]
y = df_processed['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Apply logistic regression model to training data
text_clf_reg = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf_reg', LogisticRegression(solver='lbfgs', C=10,
                                                        max_iter=1000, n_jobs=2))])
text_clf_reg.fit(x_train, y_train)
print("Training Accuracy (Logistic Regression): {}".format(text_clf_reg.score(x_train, y_train)))
predicted_reg = text_clf_reg.predict(x_test)
print("Testing Accuracy (Logdf_processedistic Regression): {}\n".format(metrics.accuracy_score(predicted_reg, y_test)))

joblib.dump(text_clf_reg, 'LR.joblib') 


"""
Function to draw validation_curve using train_scores, test_scores, 
and range of C parameter
"""
def draw_validation_curve(train_scores, test_scores, C_param_range):
    train_mean = np.mean(train_scores,axis=1)
    train_std = np.std(train_scores,axis=1)
    test_mean = np.mean(test_scores,axis=1)
    test_std = np.std(test_scores,axis=1)

    plt.figure(figsize=(15, 10))
    plt.subplot(2,2,2)
    plt.plot(C_param_range
                ,train_mean
                ,color='blue'
                ,marker='o'
                ,markersize=5
                ,label='training accuracy')

    plt.plot(C_param_range
                ,test_mean
                ,color='green'
                ,marker='x'
                ,markersize=5
                ,label='test accuracy') 

    plt.xlabel('C_parameter')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.5,1])
    plt.show()


C_param_range = [0.1,1,10,100,500,1000,2000,3000,4000,5000]
train_scores_logistic, test_scores_logistic = validation_curve(estimator=text_clf_reg
                                                            ,X=x
                                                            ,y=y
                                                            ,param_name='clf_reg__C'
                                                            ,param_range=C_param_range)
draw_validation_curve(train_scores_logistic, test_scores_logistic, C_param_range)

