"""
May 20, 2024

This file contains code implementing estimation of the odds ratio and the ACE. We estimate
the ACE using a two-stage linear regression estimator. 

This file also implements generating the semi-synthetic dataset and training bag of words 
classifiers for diagnosing signal in the text data.
"""

import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
from scipy.special import expit
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import json 
import pprint 
import itertools
from backdoor import *

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def proximal_baseline(A, Y, covariates, data):
    """
    calculate baseline proximal causal inference estimates when the proxies W and Z 
    are guanranteed to be valid proxies
    """
    # add the valid proxies to the dataframe
    data['W'] = np.random.binomial(1, expit(data['U']), len(data))
    data['Z'] = np.random.binomial(1, expit(data['U']), len(data))
    
    # print metrics for the random baseline
    metrics = all_estimate_metrics(data, hasBootstrap=True, verbose=False,
                                        confounders=covariates)

    return metrics

def random_baseline(A, Y, covariates, data, p=0.5):
    """
    calculate baseline proximal causal inference estimates when the proxies W and Z 
    are just random values
    """
    # create random baseline values
    W_base = np.random.binomial(1, p, len(data))
    Z_base = np.random.binomial(1, p, len(data))

    # add the values to the dataframe
    data['W'] = W_base
    data['Z'] = Z_base
    
    # print metrics for the random baseline
    metrics = all_estimate_metrics(data, hasBootstrap=True, verbose=False,
                                        confounders=covariates)

    return metrics

def backdoor_baseline(A, Y, W, covariates, data):
    """
    calculate baseline when using W directly in backdoor adjustment
    """
    backdoor_base = {}

    backdoor_base['ace'] = backdoor_adjustment(Y, A, [W]+covariates, data)
    backdoor_base['ci'] = compute_confidence_intervals_backdoor(Y, A, [W]+covariates, data, 'backdoor')

    return backdoor_base

def odds_ratio(X, Y, Z, data):
    features = data[[Y]+Z]
    outcome = data[X]

    if np.mean(outcome) < 0.2 or np.mean(outcome) > 0.8:
        model = LogisticRegression(class_weight="balanced", penalty=None)
    else:
        model = LogisticRegression(penalty=None)
    model.fit(features, outcome)

    return np.exp(model.coef_[0][0])

def odds_ratio_confidence_interval(X, Y, Z, data, num_bootstraps=200, alpha=0.05):
    """
    Get bootstrap confidence intervals for the value of the odds ratio
    """

    Ql = alpha/2
    Qu = 1 - alpha/2
    # two lists for the two indexes of output
    estimates = []
    
    for i in range(num_bootstraps):
        
        # resample the data with replacement
        data_sampled = data.sample(len(data), replace=True)
        data_sampled.reset_index(drop=True, inplace=True)
        
        # add estimate from resampled data
        output = odds_ratio(X, Y, Z, data_sampled)
        estimates.append(output)

    # calculate the quantiles
    quantiles = np.quantile(estimates, q=[Ql, Qu])
    q_low = quantiles[0]
    q_up = quantiles[1]
    
    return (q_low, q_up)

def proximal_find_ace(A, Y, W, Z, covariates, data):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=[Y]), data[Y], test_size=0.5)

    # subset the features to just A, Z, and covariates and the outcome to W
    model1_features = X_train[[A]+[Z]+covariates]
    model1_outcome = X_train[W]

    # if there is a high amount of class imbalance, rebalance the class weights
    if np.mean(model1_outcome) < 0.2 or np.mean(model1_outcome) > 0.8:
        model1 = LogisticRegression(class_weight="balanced", penalty=None)
    else:
        model1 = LogisticRegression(penalty=None)

    model1.fit(model1_features, model1_outcome)

    # make predictions on the probability
    What = model1.predict_proba(X_test[[A]+[Z]+covariates])[:, 1]
    # print(np.mean(What))

    X_test["What"] = What

    # train a linear regression for the second stage of the estimation strategy
    model2_features = X_test[[A]+["What"]+covariates]
    model2_outcome = y_test

    model2 = LinearRegression()
    model2.fit(model2_features, model2_outcome)
    
    return model2.coef_[0]

def compute_confidence_intervals(A, Y, W, Z, covariates, data, num_bootstraps=200, alpha=0.05):
    """
    Compute confidence intervals for proximal causal inference via bootstrap
    
    Returns tuple (q_low, q_up) for the lower and upper quantiles of the confidence interval.
    """
    
    Ql = alpha/2
    Qu = 1 - alpha/2
    # two lists for the two indexes of output
    estimates = []
    
    for i in range(num_bootstraps):
        
        # resample the data with replacement
        data_sampled = data.sample(len(data), replace=True)
        data_sampled.reset_index(drop=True, inplace=True)
        
        # add estimate from resampled data
        output = proximal_find_ace(A, Y, W, Z, covariates, data_sampled)
        estimates.append(output)

    # calculate the quantiles
    quantiles = np.quantile(estimates, q=[Ql, Qu])
    q_low = quantiles[0]
    q_up = quantiles[1]
    
    return (q_low, q_up)

def generate_semi_synthetic(data):
    semi_synthetic = {}
    # Set the synthetic seed 
    rng = np.random.default_rng(1)
    np.random.seed(0)

    size = len(data)

    # standardize the value of age since it is a large variable
    data['age'] = (data['age'] - np.mean(data['age']))/np.std(data['age'])

    """
    notes of what coefficients work well
    U: 1.425 both
    Cs: 0.9 everywhere
    """
    # A = np.random.binomial(1, expit(0.8*data['U'] + 0.8*data['gender'] + 0.8*(data['age'] - 67)), size)
    A = np.random.binomial(1, expit(1*data['U'] + 0.9*data['gender'] + 0.9*data['age']), size)
    # Y = np.random.normal(0, 1, size) + 1.3*A + 1.4*data['U'] + 0.8*data['gender'] + 0.5*data['age']
    Y = np.random.normal(0, 1, size) + 1.3*A + 1*data['U'] + 0.9*data['gender'] + 0.9*data['age']

    #Parts that are real
    semi_synthetic['C1'] = data['gender']
    semi_synthetic['C2'] = data['age']
    semi_synthetic['U'] = data['U']
    semi_synthetic['W'] = data['W']
    semi_synthetic['Z'] = data['Z']
    semi_synthetic['HADM_ID'] = data['HADM_ID']
    
    # Parts that are synthetic 
    semi_synthetic['A'] = A
    semi_synthetic['Y'] = Y
    return pd.DataFrame(semi_synthetic)

def load_W(W_config, verbose=True):
    """
    W_config is a dictionary with two keys: 'U_full_name' and 'note_category'
    W_config['U_full_name'] contains the full name of the diagnosis as specified in the jsonl file
    W_config['note_category'] contains the type of note, choose from: ECG, Echo, Nursing, Radiology
    W_config['model'] contains the type of model, choose from: flan, olmo

    model is a string, must be either 'flan' or 'olmo'
    """
    U_full_name = W_config['U_full_name']
    Wdata = [] 
    
    fname = f'{W_config['model']}_jsonl_files/{W_config['model']}--'+W_config['note_category']+'.jsonl'
    
    if verbose: print('reading in: ', fname)
    with open(fname, 'r') as r: 
        for line in r: 
            dd = json.loads(line)
            Wdata.append(dd)
    W_df = pd.DataFrame(Wdata)

    if verbose: print('\t subsetting to:'+U_full_name)
    W_df = W_df[['HADM_ID', U_full_name]].rename(columns={U_full_name: U_full_name+'_'+W_config['note_category']})

    return W_df 

def combine_dataframe(W_config, Z_config, cur_U, diagnoses_df, synthetic_Z_flag=False):
    baselines = pd.read_csv('baselines.csv')

    # load the Flan predictions for W
    W = load_W(W_config, verbose=False)

    # load the Flan predictions for Z
    Z = load_W(Z_config, verbose=False)

    # merge the predictions and only keep the intersection of hospital admissions from the Flan predictions
    combined_df = W.merge(Z, how='inner', on='HADM_ID')

    # merge the dataframe with the ground truth U's
    combined_df = combined_df.merge(diagnoses_df[['HADM_ID', 'SUBJECT_ID', cur_U]], how='left', on='HADM_ID')

    # rename columns to format used to estimate metrics
    combined_df = combined_df.rename(columns={cur_U+'_'+W_config['note_category']: 'W', 
                                              cur_U+'_'+Z_config['note_category']: 'Z',
                                              cur_U: 'U'})
    
    # add baseline information age and gender
    combined_df = combined_df.merge(baselines[['HADM_ID', 'age', 'gender']], how='left', on='HADM_ID')

    # generate semi-synthetic dataset
    semi_syn = generate_semi_synthetic(combined_df)

    # add the rest of the diagnoses as additional confounders in the semi-synthetic dataset
    new_confounders = diagnoses_df.drop(columns=['SUBJECT_ID', cur_U])

    # set up a dictionary for renaming the columns of the diagnoses
    diagnoses = list(new_confounders.columns)
    # we don't want to rename the HADM_ID column, so remove it now
    diagnoses.remove('HADM_ID')

    rename_dict = {}
    count = 3

    for diagnosis in list(diagnoses):
        rename_dict[diagnosis] = 'C'+str(count)
        count += 1

    # rename the diagnoses to C1, C2, ...
    new_confounders = new_confounders.rename(columns=rename_dict)

    # merge the new confounders onto the semi-synthetic dataset
    semi_syn = semi_syn.merge(new_confounders, how='left', on='HADM_ID')

    if synthetic_Z_flag:
        # rewrite Z to be a fully synthetic variable that is a function of just U
        semi_syn['Z'] = np.random.binomial(1, expit(semi_syn['U']), len(semi_syn))

    return semi_syn

def all_estimate_metrics(df_join, hasBootstrap=False, verbose=True, confounders=['C1', 'C2']): 
    all_metrics = {}
    est_metrics = {}

    est_metrics['gamma WZ.UC']= odds_ratio('W', 'Z', ['U']+confounders, df_join)
    est_metrics['gamma WZ.C'] =  odds_ratio('W', 'Z', confounders, df_join)
    est_metrics['ace'] = proximal_find_ace('A', 'Y', 'W', 'Z', confounders, df_join)
    est_metrics['WZ agreement'] = np.mean(df_join['W'] == df_join['Z'])

    if hasBootstrap:
        est_metrics['ci'] = compute_confidence_intervals('A', 'Y', 'W', 'Z', confounders, df_join, num_bootstraps=200, alpha=0.05) 
        est_metrics['or_ci'] = odds_ratio_confidence_interval('W', 'Z', confounders, df_join, num_bootstraps=200, alpha=0.05)
    
    for k, v in est_metrics.items(): 
        if verbose: print("{0:<30} {1:<40}".format(k, v))
    all_metrics['est_metrics'] = est_metrics


    # Other metrics with Z and W 
    U_arr = df_join['U'].to_numpy()
    Z_arr = df_join['Z'].to_numpy()
    W_arr = df_join['W'].to_numpy()

    Z_metrics = {}
    Z_metrics['precision'] = precision_score(U_arr, Z_arr)
    Z_metrics['recall'] = recall_score(U_arr, Z_arr)
    Z_metrics['accuracy'] = accuracy_score(U_arr, Z_arr)
    Z_metrics['P(Z=1)'] = np.mean(Z_arr)
    Z_metrics['gamma ZU.C'] = odds_ratio('Z', 'U', confounders, df_join)
    if verbose: 
        print("Z metrics")
        pprint.pprint(Z_metrics)
    all_metrics['Z_metrics'] = Z_metrics


    W_metrics = {}
    W_metrics['precision'] = precision_score(U_arr, W_arr)
    W_metrics['recall'] = recall_score(U_arr, W_arr)
    W_metrics['accuracy'] = accuracy_score(U_arr, W_arr)
    W_metrics['P(W=1)'] = np.mean(W_arr)
    W_metrics['gamma WU.C'] = odds_ratio('W', 'U', confounders, df_join)
    if verbose:
        print("W metrics")
        pprint.pprint(W_metrics)
    all_metrics['W_metrics'] = W_metrics
    
    return all_metrics

def join_U_text(U_text, text_piece, semisyn, text_df):
    """
    U_text: e.g., 'afib'
    text_piece: e.g., 'notes' or 'split1_onConclusions'
    """
    

    df1 = semisyn[U_text]
    df_join = df1.merge(text_df, on='patient_id')
    df_join = df_join[['U', text_piece]]
    return df_join

def combine_notes(df, identifier, note_type):
    # combine all of the notes for the same patient into one long string

    combined_notes = pd.DataFrame({'SUBJECT_ID': [], 'HADM_ID': [], note_type: []})

    unique_ids = list(set(df[identifier]))

    for id in unique_ids:
        df_id = df[df[identifier] == id]

        notes = df_id['TEXT']

        # get the subject ID corresponding to this hospital admission
        subject_id = df_id.iloc[0]['SUBJECT_ID']

        appended_text = ''
        for note in notes:
            appended_text += note

        new_row = pd.DataFrame({'SUBJECT_ID': [subject_id], 'HADM_ID': [id], note_type: [appended_text]})

        combined_notes = pd.concat([combined_notes, new_row], ignore_index=True)

    return combined_notes

def train_predict(df_join, U_text, text_piece, stop_words=None, vocabulary=None, max_iter=1000):
    """
    Train and predict 

    LR, bag of words 
    """
    VOCAB_SIZE = 5000
    #MAX_ITER = 1000

    # Fit Logistic Regression bag of words
    # Convert text to BoW features
    vectorizer = CountVectorizer(max_features=VOCAB_SIZE, stop_words=stop_words, vocabulary=vocabulary)
    X = vectorizer.fit_transform(df_join[text_piece])
    y = df_join[U_text].to_numpy()

    # Train a Logistic Regression classifier
    classifier = LogisticRegression(penalty='none', max_iter=max_iter)
    classifier.fit(X, y)

    # Make predictions
    y_pred = classifier.predict(X)

    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred) 
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    return {'U': U_text, 
            'text': text_piece, 
            'num_rows': df_join.shape[0], 
            'accuracy': accuracy, 
            'precision': precision,
            'recall': recall, 
            'f1': f1,
            'E[y_hat]': np.mean(y_pred),
            'p(U)=1': np.mean(df_join[U_text]) 
            }, classifier, vectorizer, y_pred

def top_words(U_text, text_piece, save, top_n=10): 
    clf, vec = save[(U_text, text_piece)]

    coefficients = clf.coef_[0]
    feature_names = vec.get_feature_names_out()
    word_importances = zip(feature_names, coefficients)
    most_important_words = sorted(word_importances, key=lambda x: abs(x[1]), reverse=True)     

    # Print the top N most important words with their coefficients
    for word, coef in most_important_words[:top_n]:
        print(f"{word}: {coef}")

