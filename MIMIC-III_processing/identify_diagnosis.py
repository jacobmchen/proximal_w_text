"""
This files provides a function for updating the master csv file to add a row for whether or not
they were diagnosed with a certain diagnosis given their ICD_9 code.
"""
import pandas as pd
import numpy as np

def identify_diagnosis(ICD_9_code):
    diagnoses = pd.read_csv('../MIMIC-III/DIAGNOSES_ICD.csv')

    # go through the csv file DIAGNOSIS_ICD, save a dictionary for each patient id
    # for whether or not they received a diagnosis of ICD_9_code
    patient_diagnosis = {}

    for index, row in diagnoses.iterrows():
        if row['ICD9_CODE'] == ICD_9_code:
            patient_diagnosis[int(row['SUBJECT_ID'])] = 1
        elif patient_diagnosis.get(int(row['SUBJECT_ID'])) == 0 or patient_diagnosis.get(int(row['SUBJECT_ID'])) == None:
            patient_diagnosis[int(row['SUBJECT_ID'])] = 0

    # go through the master csv, for each row check if in the dictionary they had a
    # diagnosis of 1
    master_csv = pd.read_csv('csv_files/master_data.csv')

    # diagnosis_status is an array that keeps track of whether each patient was diagnosed with
    # the thing indicated by the ICD9 code
    diagnosis_status = []
    for index, row in master_csv.iterrows():
        diagnosis_status.append(patient_diagnosis[int(row['patient_id'])])

    # return the diagnosis_status that corresponds to each patient's diagnosis status
    # the order is the same as the order of patients in master_csv
    return diagnosis_status 
