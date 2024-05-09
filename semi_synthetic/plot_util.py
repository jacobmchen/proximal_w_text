import pandas as pd 
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def or_test(or_low, or_high, gamma_min, gamma_max):
    """
    Given an odds ratio confidence interval range and minimum and maximum values for the range,
    return a string that indicates whether the confidence interval range is within the minimum
    and maximum values
    """
    if or_low > gamma_min and or_high < gamma_max:
        return 'Passed'
    else:
        return 'Failed'

def create_csv(input_file, output_file, gamma_min=1, gamma_max=5):
    """
    This function takes the pickle file output from estimate_grid.ipynb and converts it
    into a csv file format we will use for creating plots.

    Each row has 10 columns: oracle, method, W_category, W_model, Z_category, Z_model, ci_mean, ci_low, ci_high,
    or_test
    """
    # read the pickle file
    estimate_grid_save = pickle.load(open(input_file, 'rb'))

    column_plot_data = pd.DataFrame({'oracle': [], 'method': [], 'W_category': [], 'W_model': [],
                                     'Z_category': [], 'Z_model': [], 'ci_mean': [], 'ci_low': [],
                                     'ci_high': [], 'or_test': []})

    for estimate in estimate_grid_save:
        row = pd.DataFrame()

        row['oracle'] = [estimate['oracle']]*2
        row['method'] = ['backdoor', 'proximal']
        row['W_category'] = [estimate['W_config']['note_category']]*2
        row['W_model'] = [estimate['W_config']['model']]*2
        row['Z_category'] = [estimate['Z_config']['note_category']]*2
        row['Z_model'] = [estimate['Z_config']['model']]*2
        row['ci_mean'] = [estimate['backdoor_baseline']['ace'], estimate['all_metrics']['est_metrics']['ace']]
        row['ci_low'] = [estimate['backdoor_baseline']['ci'][0], estimate['all_metrics']['est_metrics']['ci'][0]]
        row['ci_high'] = [estimate['backdoor_baseline']['ci'][1], estimate['all_metrics']['est_metrics']['ci'][1]]
        row['or_test'] = ['Neutral', 
                          or_test(estimate['all_metrics']['est_metrics']['or_ci'][0], estimate['all_metrics']['est_metrics']['or_ci'][1],
                                                        gamma_min, gamma_max)]
        
        # metrics needed for table 2
        row['gamma_WZ_UC'] = [None, estimate['all_metrics']['est_metrics']['gamma WZ.UC']]
        row['or_ci_low'] = [None, estimate['all_metrics']['est_metrics']['or_ci'][0]]
        row['or_ci_high'] = [None, estimate['all_metrics']['est_metrics']['or_ci'][1]]

        # metrics needed for appendix G
        row['oracle_positivity'] = [estimate['oracle_positivity']]*2

        row['gamma WU.C'] = [None, estimate['all_metrics']['W_metrics']['gamma WU.C']]
        row['W_accuracy'] = [None, estimate['all_metrics']['W_metrics']['accuracy']]
        row['p(W=1)'] = [None, estimate['all_metrics']['W_metrics']['P(W=1)']]
        row['W precision'] = [None, estimate['all_metrics']['W_metrics']['precision']]
        row['W recall'] = [None, estimate['all_metrics']['W_metrics']['recall']]

        row['gamma ZU.C'] = [None, estimate['all_metrics']['Z_metrics']['gamma ZU.C']]
        row['Z_accuracy'] = [None, estimate['all_metrics']['Z_metrics']['accuracy']]
        row['p(Z=1)'] = [None, estimate['all_metrics']['Z_metrics']['P(Z=1)']]
        row['Z precision'] = [None, estimate['all_metrics']['Z_metrics']['precision']]
        row['Z recall'] = [None, estimate['all_metrics']['Z_metrics']['recall']]

        row['WZ agreement'] = [None, estimate['all_metrics']['est_metrics']['WZ agreement']]

        column_plot_data = pd.concat([column_plot_data, row], ignore_index=True)

    # in the method column, rename proximal to either P1M, P2M_fo (flan for W and olmo for Z), P2M_of (olmo for W and flan for Z)
    column_plot_data['method'] = column_plot_data.apply(lambda row: row['method'] if row['method'] != 'proximal' else ('proximal_P1M' if row['W_model'] == row['Z_model'] else ('proximal_P2M_fo' if row['W_model'] == 'flan' else 'proximal_P2M_of')), axis=1)

    # drop repeated rows in the dataframe for baseline and backdoor since the baseline and backdoor metrics are
    # recalculated a couple of times from calculating P1M and P2M calculations having the same baselines
    column_plot_data = column_plot_data.drop_duplicates(subset=['oracle', 'method', 'W_category', 'Z_category'], ignore_index=True)

    # replace flan-olmo and olmo-flan identifiers with just P2M
    column_plot_data = column_plot_data.replace({'proximal_P2M_fo': 'proximal_P2M', 'proximal_P2M_of': 'proximal_P2M'})

    # save dataframe to a csv file
    column_plot_data.to_csv(output_file, index=False)
