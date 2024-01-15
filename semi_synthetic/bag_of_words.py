from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
import pandas as pd
import numpy as np
import re

def create_bow_model(max_words=None, ngram_range=(1,1)):
    if max_words == None:
        text_clf = Pipeline([
            ('vect', CountVectorizer(ngram_range=ngram_range)),
            ('tfidf', TfidfTransformer()),
            ('lr', LogisticRegression(C=1e20)),
        ])
        return text_clf
    else:
        text_clf = Pipeline([
            ('vect', CountVectorizer(max_features=max_words, ngram_range=ngram_range)),
            ('tfidf', TfidfTransformer()),
            ('lr', LogisticRegression(C=1e20)),
        ])
        return text_clf

def create_confusion_matrix(truth, predictions):
    # calculate the confusion matrix
    tn, fp, fn, tp = confusion_matrix(truth, predictions).ravel()

    output = {}

    output['tn'] = tn
    output['fp'] = fp
    output['fn'] = fn
    output['tp'] = tp

    # calculate the sensitivity and specificty of the data
    sensitivity = tp / (tp+fn)
    specificity = tn / (fp+tn)
    output['sensitivity'] = sensitivity
    output['specificity'] = specificity

    # calculate precision and recall
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    output['precision'] = precision
    output['recall'] = recall

    return output