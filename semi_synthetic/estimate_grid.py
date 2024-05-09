import json
import pandas as pd
import numpy as np
import re
from estimation_util import *
from backdoor import *
from plot_util import *
import pickle

np.random.seed(0)

original_diagnoses_df = pd.read_csv('diagnoses_df.csv')
# drop the column indicating 'newborn suspected infection'
diagnoses_df = original_diagnoses_df.drop(columns=['NB obsrv suspct infect'])
# rename the relevant columns with long names
diagnoses_df = diagnoses_df.rename(columns={'Acute respiratry failure': 'acute respiratory failure',
                                            'Atrial fibrillation': 'atrial fibrillation',
                                            'CHF NOS': 'congestive heart failure',
                                            'Crnry athrscl natve vssl': 'coronary atherosclerosis of native coronary artery',
                                            'Hypertension NOS': 'hypertension'})

# these are the mappings for the truncated notes, only remove acute respiratory failure
diagnosis_note_category_map = {'atrial fibrillation1': ('ECG', 'Echo'),
                               'atrial fibrillation2': ('ECG', 'Nursing'),
                               'atrial fibrillation3': ('Echo', 'Nursing'),
                               'congestive heart failure': ('Echo', 'Nursing'),
                               'coronary atherosclerosis of native coronary artery1': ('Echo', 'Nursing'),
                               'coronary atherosclerosis of native coronary artery2': ('Echo', 'Radiology'),
                               'coronary atherosclerosis of native coronary artery3': ('Radiology', 'Nursing'),
                               'hypertension': ('Echo', 'Nursing')}

diagnosis_name_map = {'atrial fibrillation1': 'A-Fib',
                        'atrial fibrillation2': 'A-Fib',
                        'atrial fibrillation3': 'A-Fib',
                        'congestive heart failure': 'Heart',
                        'coronary atherosclerosis of native coronary artery1': 'A-Sis',
                        'coronary atherosclerosis of native coronary artery2': 'A-Sis',
                        'coronary atherosclerosis of native coronary artery3': 'A-Sis',
                        'hypertension': 'Hypertension'}

# use all of the baseline covariates
# C1 and C2 are age and gender, respectively
baseline_covariates = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']

# some have multiple numbers because there are multiple possible note combinations that we want to try
U_LONG = ['atrial fibrillation1', 'atrial fibrillation2', 'atrial fibrillation3', 'congestive heart failure',
          'coronary atherosclerosis of native coronary artery1', 'coronary atherosclerosis of native coronary artery2',
          'coronary atherosclerosis of native coronary artery3', 'hypertension']

# # only run the examples that I want to plot
# U_LONG = ['coronary atherosclerosis of native coronary artery2', 'congestive heart failure']

# MODEL_CHOICES = [('flan', 'flan'), ('olmo', 'olmo'), ('flan', 'olmo'), ('olmo', 'flan')]
# ignore P1M olmo-olmo
MODEL_CHOICES = [('flan', 'flan'), ('flan', 'olmo'), ('olmo', 'flan')]

# declare a list to save all of the outputs into, each element is a dictionary with the following keys:
# ['W_config'] = the dictionary W_config
# ['Z_config'] = the dictionary Z_config
# ['all_metrics'] = the dictionary containing all the estimation metrics ['W_metrics'], ['Z_metrics'], and ['est_metrics']
# ['W_Z_baseline'] = the dictionary containing baseline metrics when W and Z are generated randomly in proximal
# ['backdoor_baseline'] = the dictionary containing baseline metrics when we use W directly in backdoor adjustment
estimate_grid_save = []

for u_index in range(len(U_LONG)):
    # make sure we try both possibilities for P2M where we use Flan for the first note category and 
    # olmo for the second note category
    # also try both possibilities for P1M where we only use flan or only use olmo predictions
    for model_choice in MODEL_CHOICES:
        print("==="*10)
        print("U=", U_LONG[u_index])

        cur_U = re.sub(r'\d+', '', U_LONG[u_index]) # use regular expression matching to remove any possible numbers

        W_config = {}
        W_config['U_full_name'] = cur_U
        W_config['note_category'] = diagnosis_note_category_map[U_LONG[u_index]][0]
        W_config['model'] = model_choice[0]

        Z_config = {}
        Z_config['U_full_name'] = cur_U
        Z_config['note_category'] = diagnosis_note_category_map[U_LONG[u_index]][1]
        Z_config['model'] = model_choice[1]

        semi_syn = combine_dataframe(W_config, Z_config, cur_U, diagnoses_df, synthetic_Z_flag=False)
        print('Y mean', np.mean(semi_syn['Y']))
        print('A mean', np.mean(semi_syn['A']))

        all_metrics = all_estimate_metrics(semi_syn, hasBootstrap=True, verbose=False,
                                        confounders=baseline_covariates)

        # print out only positive for now 
        ace = all_metrics['est_metrics']['ace']
        oracle_or = all_metrics['est_metrics']['gamma WZ.UC']
        observed_or = all_metrics['est_metrics']['gamma WZ.C']
        print('n=', len(semi_syn))
        print('W category=', W_config['note_category'])
        print('W model=', W_config['model'])
        print('Z category=', Z_config['note_category'])
        print('Z model=', Z_config['model'])

        print('U positivity rate=', np.mean(semi_syn['U']))
        print('ace=', ace)
        print('gamma WZ.UC=', oracle_or)
        print('gamma WZ.C=', observed_or)
        print('W_metrics', all_metrics['W_metrics'])
        print('Z_metrics', all_metrics['Z_metrics'])

        pprint.pprint(all_metrics)

        W_backdoor = backdoor_baseline('A', 'Y', 'W', baseline_covariates, semi_syn)
        print('backdoor metrics')
        pprint.pprint(W_backdoor)

        # # sanity check: can backdoor with access to the oracle recover the ACE?
        # real_backdoor = backdoor_baseline('A', 'Y', 'U', baseline_covariates, semi_syn)
        # print('real backdoor')
        # pprint.pprint(real_backdoor)

        # sanity check: can W being flan predictions and Z being synthetic recover the ACE
        # in proximal?
        semi_syn = combine_dataframe(W_config, Z_config, cur_U, diagnoses_df, synthetic_Z_flag=True)
        W_flan_Z_synthetic = all_estimate_metrics(semi_syn, hasBootstrap=True, verbose=False,
                                        confounders=baseline_covariates)
        pprint.pprint(W_flan_Z_synthetic)

        # sanity check: can two synthetic proxies that are guanranteed to satisfy identification
        # conditions recover the true ACE?
        proximal = proximal_baseline('A', 'Y', baseline_covariates, semi_syn)
        print('real proximal')
        pprint.pprint(proximal)

        estimate_grid_save.append({'oracle_positivity': np.mean(semi_syn['U']),
                                   'oracle': diagnosis_name_map[U_LONG[u_index]], 'W_config': W_config, 'Z_config': Z_config,
                                   'all_metrics': all_metrics, 'backdoor_baseline': W_backdoor, 
                                   'W_flan_Z_synthetic': W_flan_Z_synthetic, 'W_synthetic_Z_synthetic': proximal})
        
file_name_addon = 'sklearn_nopenalty'

pickle.dump(estimate_grid_save, open(f'estimate_grid_outputs/estimate_grid_save_{file_name_addon}.p', 'wb'))