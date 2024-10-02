# Procedure Outline

May 8, 2024

This document outlines the procedures for getting the final proximal estimator.

Useful links:
- Official MIMIC-III documentation: https://mimic.mit.edu/.
- Liu, Jiewen, et al. "Regression-Based Proximal Causal Inference." arXiv preprint arXiv:2402.00335 (2024): https://arxiv.org/abs/2402.00335.

## Instructions for Running Code

Assuming the reader has access to the MIMIC-III dataset, the reader may reproduce our semi-synthetic experiments in the following steps:

1. Run `preprocessing.ipynb` in the folder `preprocessing`.
2. Run `diagnose_f1_scores.py` in the folder `diagnose_f1_scores`. If creating LaTex tables to summarize results, run `create_latex_tables.ipynb`.
3. Run `flan_infer_proxy.py` and `olmo_infer_proxy.py` to make inferences from text data.
4. Run `estimate_grid.py` to execute semi-synthetic experiments.
5. Run `ci_plot.ipynb` and `create_latex_tables.py` to create plots and LaTeX tables summarizing results from the semi-synthetic experiments.

## Descriptions of Key Files

### Preprocessing Steps

1. Read NOTEEVENTS.csv from MIMIC-III. Drop rows of data where "HADM_ID" has a value of nan.
2. Subset NOTEEVENTS.csv by choosing a "HADM_ID" for each "SUBJECT_ID" by selection on lowest/earliest value of "CHARTDATE".
3. Create a dataframe output_A.csv by taking the previously created subset and dropping all notes that are labeled as "Discharge summary". Keep "CATEGORY" as a column.
4. Read DIAGNOSES_ICD.csv and D_ICD_DIAGNOSES.csv from MIMIC-III. Create a dataframe output_B.csv that maps the "SHORT_TITLE" of each ICD9 code to DIAGNOSES_ICD.csv. Basically, we are just mapping the short titles to the ICD9 codes diagnosed in DIAGNOSES_ICD.csv.
5. Iterate through each row in output_B.csv. For each "HADM_ID" value in each row, check if it is one of the "HADM_ID"s in output_A.csv. If it is, record in a Counter() object that the diagnosis for that row appeared.
6. Look at the top 10 most commonly appearing diagnoses for the "HADM_ID" values in output_A.csv. The short tiles are:
    1. Hypertension NOS (hypertension, not otherwise specified)
    2. Crnry athrscl natve vssl (coronary atherosclerosis of native coronary artery, buildup of fats, cholesterol, and other substances in the blood)
    3. Atrial fibrillation
    4. CHF NOS (congestive heart failure, not otherwise specified)
    5. DMII wo cmp nt st uncntr (diabetes mellitus without mention of complication)
    6. Hyperlipidemia NEC/NOS (hyperlipidemia, high levels of fats in the blood)
    7. Acute kidney failure NOS (acute kidney failure, not otherwise specified)
    8. Need prphyl vc vrl hepat (need vaccination and inoculation against viral hepatitis)
    9. NB obsrv suspct infect (newborn suspected infection)
    10. Acute respiratry failure
7. Create a dataframe diagnoses_df.csv that records for each "HADM_ID" whether the patient was diagnosed with one of the top 10 diagnoses. We do so by first creating a dataframe with all the unique "HADM_ID"s in output_A.csv. For each unique "HADM_ID", we check output_B if that "HADM_ID" had an entry for one of the top 10 diagnoses. If yes, then diagnoses_df.csv records a value of 1 for that diagnosis. Otherwise, diagnoses_df.csv records a value of 0.
8. Record the basline confounders gender and age from PATIENTS.csv. We do so by mapping the values in the column "GENDER" from 'F' to 0 and 'M' to 1. Then, we merge the column "GENDER" to output_A.csv on "SUBJECT_ID" (We merge on "SUBJECT_ID" because PATIENTS.csv does not contain the column "HADM_ID"). We then merge the column "DOB" (date of birth) to output_A.csv and use the column "CHARTDATE" to infer the patient's age. We drop patients that are younger than 18 or older than 100.
9. As the final preprocessing step, we investigate which note categories appear the most often. Initially, there are 14 note categories:
    1. Case Management
    2. Consult
    3. ECG
    4. Echo
    5. General
    6. Nursing
    7. Nursing/other
    8. Nutrition
    9. Pharmacy
    10. Physician
    11. Radiology
    12. Rehab Services
    13. Respiratory
    14. Social Work
    
    We first create a dictionary where the key is a "HADM_ID" and the value is a list containing all of the note categories available for that patient for that "HADM_ID". We then iterate through all possible combinations of choosing 2 of the note categories (14 choose 2 is 91 possible combinations) and check how many "HADM_ID"s in the previously created dictionary possess both of the note categories in its list. The five most common combinations with the amount of "HADM_ID"s/patients were:
    1. ECG and Radiology: 27963 patients
    2. Nursing/other and ECG: 18505 patients
    3. Nursing/other and Radiology: 18302 patients
    4. ECG and Echo: 17291 patients
    5. Echo and Radiology: 15558 patients

    There are 4 note categories appearing in these top 5 pairings: ECG, Echo, Nursing/other, and Radiology. We now only consider these 4 note categories. Note that ECG (electrocardiogram) and Echo (echocardiogram) are distinct types of clinician's notes.

    *Note: From now on, we will refer to the note category "Nursing/other" as simply "Nursing".

All of these steps are implemented in the file preprocessing.ipynb in the preprocessing folder.

### Diagnosing Signal

To determine which notes actually had any signal for potential diagnoses in a semi-synthetic setup, we ran simple bag of words (BoW) linear logistic regressions on the previously identified note categories and the oracle diagnoses. These experiments are contained in the file diagnose_f1_scores.py in the diagnose_f1_scores folder.

1. Because some "HADM_ID"s had multiple notes from the same category, we first combined all of the notes of the same category for each "HADM_ID" into one long string. We truncate the note to the first 470 tokens (the reason for taking this preprocessing step is explained below).
    1. ***Note:*** After running the pipeline outlined in this document once, we realized that Flan is not capable of accepting inputs with more than 512 tokens. Given that we concatenated all of the notes together in the step above, many notes ended up exceeding the maximum length. This caused Flan to skip notes for many patients and hospital admissions. To mitigate this, we truncated all of the merged text data to 470 tokens so that the total length of the string, including the prompt, provided as input to Flan would not exceed 512 tokens.
2. For each of the 4 note categories, we trained a BoW linear logistic regression on each of the 10 possible diagnoses. An important intermediate step that we took was to disregard any oracle diagnoses that only had one class after subsetting to the note category of interest. We do this because supervised learning tasks are not useful when there is only one class to predict.

    In total, we trained 40 BoW linear logistic regressions. The full results are saved in the pickle files "diagnose_f1_scores_save.p" and "diagnose_f1_scores_all_results.p". 
3. The linear logistic regressions with F1 scores greater than 0.7 were inspected and saved to a separate file 'top_f1_scores.txt'.
4. Based on the results, we will use the following oracle diagnoses and note categories to infer proxies with in the semi-synthetic experiments:

    1. atrial fibrillation: ECG and Echo notes
    2. atrial fibrillation: ECG and Nursing notes
    3. atrial fibrillation: Echo and Nursing notes
    4. congestive heart failure: Echo and Nursing notes
    5. coronary atherosclerosis of native coronary artery: Echo and Nursing notes
    6. coronary atherosclerosis of native coronary artery: Echo and Radiology notes
    7. coronary atherosclerosis of native coronary artery: Radiology and Nursing notes
    8. hypertension: Echo and Nursing notes

    Of the 8 possible setups, there are 5 distinct "oracle" diagnoses.

### Creating Flan Predictions

We ask Flan to make predictions from the 4 note categories on each of the 5 diagnoses chosen in the Diagnosing Signal section. We use the following string as a prompt for the zero-shot classifier:

'Context: ' + [note data] + '\nIs it likely the patient has '+ [full name of the diagnosis] +'?\nConstraint: Even if you are uncertain, you must pick either “Yes” or “No” without using any other words.'

If the string 'yes' appears anywhere in the lowercase version of the Flan output, we record an inference of 1 and 0 otherwise. We save the inferences from Flan in jsonl files for use in the semi-synthetic experiments.

The full code for making Flan predictions can be found in flan_infer_proxy.py.

### Creating OLMo Predictions

We use the same procedure as above to create predictions from OLMo.

OLMo's output format, however, is slightly different from that of Flan. Instead of just outputting the response to the text data and the prompt, OLMo will respond with the entirety of the input plus its response. Hence, we find the index of the string 'Constraint: Even if you are uncertain, you must pick either “Yes” or “No” without using any other words.' in the output of OLMo and only consider characters that appear after that string as the output for OLMo. As before, if the string 'yes' appears anywhere in the lowercase version of the OLMo output, we record an instance of 1 and 0 otherwise.

The full code for making OLMo predictions can be found in olmo_infer_proxy.py.

### Semi-Synthetic Experiments and Proximal

First, we construct the dataframe we will use to generate synthetic variables from. We first merge the predictions for the diagnosis from the two note categories and Flan into one dataframe. Next, we merge the ground truth "oracle" values of the diagnosis into the dataframe. Then, we merge in the baseline covariates gender and age.

Now, we generate synthetic variables (1) binary treatment variable A and (2) continuous outcome variable Y as functions of the oracle diagnosis, age, and gender. Finally, we add the rest of the diagnoses except for "NB obsrv suspct infect" (because it is a condition affecting newborns) as baseline confounders to the dataframe. This makes for a total of 8 extra confounders in addition to age and gender and excluding the diagnosis being used as the unmeasured confounder. These 8 extra confounders are not used in generating the synthetic variables.

Using the new dataframe, we use the two-stage linear regression proximal causal inference estimator for continuous outcomes and binary proxies described in Liu et al., 2024 to make predictions for the average causal effect (ACE). We do modify the procedure with the two following additions:

1. In the first stage of the linear regression proximal causal inference estimator, we use a linear logistic regression with the sklearn library to predict one of the proxies (W) as a function of the treatment, the other proxy, and baseline covariates. However, whenever there is a class imbalance, i.e. the average of W is is less than 0.2 or greater than 0.8, we reweight the dataset by the inverse probability of observing values of W. We apply the same reweighting strategy to estimate the odds ratio as well. Whenever we use a logistic regression, we set the parameter penalty='none' to pevent the model from using regularization. We don't need to do this for linear regression because the sklearn implementation of linear regression does not have regularization to begin with.
2. In the first stage of the linear regression proximal causal inference estimator, we split the dataset into two 50% splits. We use the split 1 to train a logistic regression that predicts one of the proxies W. We then use split 2 to make probabilistic predictions of W and train a linear regression for the outcome Y as a function of the treatment, probabilistic predictions of W, and baseline covariates. The coefficient of the treatment in the trained linear model is the estimate for the ACE.

We use bootstrapping to make estimates for the confidence intervals. We iterate over different possibilities for P1M Flan, P1M OLMo, and P2M Flan+OLMo while using different combinations of note categories as the proxies for proximal causal inference.

The full code for this section can be found in estimate_grid.py.