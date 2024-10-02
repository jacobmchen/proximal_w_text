"""
May 20, 2024

This file contains code using the matplotlib library to plot results from the semi-synthetic
experiments.
"""

import pandas as pd 
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_cur_axs(axs, rows, row, col):
    if rows == 1:
        return axs[col]
    else:
        return axs[row, col]

def create_plots(csv_filename, data_order, true_ace=1.3, rows=2, cols=2, width=8, height=6):
    """
    Create 2x2 bar graphs based on the data_order specified. It is possible to specify different dimensions for the
    bar graph, but we keep it at 2x2 for now.

    data_order is a list that contains four dictionaries. Each dictionary should contain 3 keys: 'oracle', 'P1M', and
    'P2M'. 'oracle' should be just a string that specifies 'A-Fib', 'Heart', 'A-Sis', or 'Hypertension'. 'P1M' and 
    'P2M' should be tuples with the following format: ('W note category', 'W model', 'Z note category', 'Z model').
    """
    # read in the csv file containing the data that we want to plot
    df = pd.read_csv(csv_filename)

    # create the labels that we're going to use in the plot
    labels = []
    for i in range(len(data_order)):
        labels.append(f'{data_order[i]['oracle']}\n{data_order[i]['P2M'][0]}/{data_order[i]['P2M'][2]}')

    data_labels_clean = ["Est. ACE for "+x for x in labels] 
    method_order = ["backdoor", "proximal_P1M", "proximal_P2M"]
    method_order_clean_labels = ["Backdoor\nw/ proxy", "P1M", "P2M"]

    fig, axs = plt.subplots(rows, cols, figsize=(width, height))

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.2) 

    ax_no = 0

    for row in range(rows):
        for col in range(cols):
            # subset to the rows of data we want to plot
            data_kind = data_order[ax_no]
            # specify the exact combination of note categories and models that you want to plot in the graph
            mask = ((df['oracle'] == data_kind['oracle']) & 
                    ( ( (df['W_model'] == data_kind['P1M'][1]) & (df['Z_model'] == data_kind['P1M'][3]) ) | ( (df['W_model'] == data_kind['P2M'][1]) & (df['Z_model'] == data_kind['P2M'][3]) ) ) &
                    ( ( (df['W_category'] == data_kind['P1M'][0]) & (df['Z_category'] == data_kind['P1M'][2]) ) | ( (df['W_category'] == data_kind['P2M'][0]) & (df['Z_category'] == data_kind['P2M'][2]) ) )
                    )
            data_subset = df[mask]
            
            #Y-label 
            get_cur_axs(axs, rows, row, col).set_ylabel(data_labels_clean[ax_no])
            
            #Put in the true value 
            get_cur_axs(axs, rows, row, col).axhline(y=true_ace, color='green', linewidth=1, label="True ACE", linestyle='--')
            
            #Other cleanup
            get_cur_axs(axs, rows, row, col).set_ylim(1.1, 1.7)
            
            for method_no in range(len(method_order)): 
                method_kind = method_order[method_no]
                method_subset = data_subset.loc[data_subset["method"] == method_kind]
                mean = method_subset["ci_mean"].iloc[0]
                lower = method_subset["ci_low"].iloc[0]
                upper = method_subset["ci_high"].iloc[0]
                error = np.array([abs(mean - lower), abs(upper - mean)]).reshape(-1, 1)
                
                OR_result = method_subset['or_test'].iloc[0]
                if OR_result == "Passed": 
                    bar_color = 'blue'
                    # Add a star if passed
                    get_cur_axs(axs, rows, row, col).text(method_no-0.05, mean, '$\checkmark$', horizontalalignment='right', verticalalignment='center', color='blue')
                elif OR_result == "Passed_nostar":
                    bar_color = 'blue'
                elif OR_result == "Failed":
                    bar_color = 'red'
                elif OR_result == "Neutral": 
                    bar_color = 'black'
                
                get_cur_axs(axs, rows, row, col).errorbar(method_no, mean, yerr=error, fmt='o', capsize=5, color=bar_color)
                
            if ax_no == 0: 
                get_cur_axs(axs, rows, row, col).legend()
                
            # Put in xtick labels for plots on the last row
            if row == rows-1:
                get_cur_axs(axs, rows, row, col).set_xticks([0, 1, 2])
                get_cur_axs(axs, rows, row, col).set_xticklabels(method_order_clean_labels)
            else: 
                get_cur_axs(axs, rows, row, col).set_xticks([])
                get_cur_axs(axs, rows, row, col).set_xticklabels([])

            ax_no += 1