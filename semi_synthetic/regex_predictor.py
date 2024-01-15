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

def regular_expression_predict(notes, expressions):
    prediction = []

    for note in notes:
        note = note.lower()

        # check if any of the phrases in expressions shows up, if neither phrase shows up then
        # we make a prediction that the patient did not experience atrial fibrillation, otherwise
        # we predict that the patient did experience atrial fibrillation

        flag = True
        for exp in expressions:
            if re.search(exp, note, flags=0) != None:
                prediction.append(1)
                flag = False
                break

        if flag:    
            prediction.append(0)

    return prediction