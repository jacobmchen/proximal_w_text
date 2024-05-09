"""
April 5, 2024
Code for testing out F1 values from a BoW logistic regression by combining different note types
and possible diagnoses.
"""

import json 
import pandas as pd
import numpy as np
import pickle
import sys
import os

# the following code lets you import a python file from a different directory
path_to_file = os.path.join(os.path.dirname('estimation_util.py'), '..')

sys.path.append(path_to_file)

from estimation_util import *

save = {}
all_results = {}

# Load data 
df = pd.read_csv("../output_A.csv")
df.shape

diagnoses_df = pd.read_csv('../diagnoses_df.csv')

### testing code ###
# note_categories = ['ECG']
# possible_diagnoses = ['Hypertension NOS']
### testing code ###

note_categories = ['ECG', 'Radiology', 'Nursing', 'Echo']
possible_diagnoses = ['Hypertension NOS', 'Crnry athrscl natve vssl', 'Atrial fibrillation', 'CHF NOS', 
                      'DMII wo cmp nt st uncntr', 'Hyperlipidemia NEC/NOS', 'Acute kidney failure NOS',
                      'Need prphyl vc vrl hepat', 'NB obsrv suspct infect', 'Acute respiratry failure']

for note_category in note_categories:
    for diagnosis in possible_diagnoses:
        """
        below code was used before we we set up preprocessing steps that saved the combined text
        data in separate files
        # # first, get all the notes data
        # specific_note_type = df[df['CATEGORY'] == note_category]

        # # second, combine all the text into one long string if there are multiple notes
        # combined_text = combine_notes(specific_note_type, 'HADM_ID', note_category)
        """

        # first, get the note data
        fname = '../text_csv_files/text_data_'+note_category+'.csv'
        combined_text = pd.read_csv(fname)

        # second, merge the full text with the proposed diagnosis
        df_join = combined_text.merge(diagnoses_df[['HADM_ID', diagnosis]], how='left', on='HADM_ID')

        # important pre-processing step: if the U is 0 or 1 for all patients, we skip
        if np.mean(df_join[diagnosis]) == 0 or np.mean(df_join[diagnosis]) == 1:
            print('For [', note_category, '] and [', diagnosis, '] data contains only one class')
            continue

        # third, train a BoW logistic regression using the combined text
        results, clf, vec, y_pred = train_predict(df_join, diagnosis, note_category)

        save[(diagnosis, note_category)] = (clf, vec)
        all_results[(diagnosis, note_category)] = results

        pprint.pprint(results)
        print()

pickle.dump(save, open('diagnose_f1_scores_save.p', 'wb'))
pickle.dump(all_results, open('diagnose_f1_scores_all_results.p', 'wb'))