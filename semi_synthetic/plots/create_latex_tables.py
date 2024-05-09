import pandas as pd

def table2(filename, data_order):
    """
    Create strings for Table 2 that displays baseline information for 
    """
    final_string = ''

    df = pd.read_csv(filename)

    for i in range(len(data_order)):
         # subset to the rows of data we want to plot
        data_kind = data_order[i]
        # specify the exact combination of note categories and models that you want to plot in the graph
        mask = ((df['oracle'] == data_kind['oracle']) & 
                ( ( (df['W_model'] == data_kind['P1M'][1]) & (df['Z_model'] == data_kind['P1M'][3]) ) | ( (df['W_model'] == data_kind['P2M'][1]) & (df['Z_model'] == data_kind['P2M'][3]) ) ) &
                ( ( (df['W_category'] == data_kind['P1M'][0]) & (df['Z_category'] == data_kind['P1M'][2]) ) | ( (df['W_category'] == data_kind['P2M'][0]) & (df['Z_category'] == data_kind['P2M'][2]) ) )
                )
        data_subset = df[mask]
        
        p1m = data_subset[data_subset['method'] == 'proximal_P1M']
        p2m = data_subset[data_subset['method'] == 'proximal_P2M']
        
        oracle = p2m['oracle'].iloc[0]
        W_model = p2m['W_category'].iloc[0]
        Z_model = p2m['Z_category'].iloc[0]
        p1m_oracle_or = p1m['gamma_WZ_UC'].iloc[0]
        p2m_oracle_or = p2m['gamma_WZ_UC'].iloc[0]
        p1m_or_low = p1m['or_ci_low'].iloc[0]
        p1m_or_high = p1m['or_ci_high'].iloc[0]
        p2m_or_low = p2m['or_ci_low'].iloc[0]
        p2m_or_high = p2m['or_ci_high'].iloc[0]

        if p1m['or_test'].iloc[0] == 'Passed':
            p1m_text_color = 'blue'
        elif p1m['or_test'].iloc[0] == 'Failed':
            p1m_text_color = 'red'
        
        if p2m['or_test'].iloc[0] == 'Passed':
            p2m_text_color = 'blue'
        elif p2m['or_test'].iloc[0] == 'Failed':
            p2m_text_color = 'red'

        final_string += f'{oracle} & {W_model} & {Z_model} & ${p1m_oracle_or:.3f}$ & ${p2m_oracle_or:.3f}$ & {{\color{{{p1m_text_color}}} $({p1m_or_low:.3f}, {p1m_or_high:.3f})$}} & {{\color{{{p2m_text_color}}} $({p2m_or_low:.3f}, {p2m_or_high:.3f})^\checkmark$}} \\\\ \n'
    
    return final_string

def sectionH(csv_filename, data_kind):
    """
    Create an entire table for the tables used in Section G of the appendix.
    """
    # create a dictionary to map short oracle names onto long oracle names
    LONG_NAMES = {
        'A-Fib': 'atrial fibrillation',
        'Heart': 'congestive heart failure',
        'A-Sis': 'coronary atherosclerosis',
        'Hypertension': 'hypertension'
    }

    # create a dictionary to map the method to text corresponding to the 
    # number of zero-shot classifiers
    METHOD_COUNT = {
        'proximal_P1M': 'one zero-shot classifier',
        'proximal_P2M': 'two zero-shot classifiers'
    }

    final_string = ''

    df = pd.read_csv(csv_filename)

    # subset to the rows of data we want to plot
    # specify the exact combination of note categories and models that you want to plot in the graph
    mask = ((df['oracle'] == data_kind['oracle']) & 
            ( ( (df['W_model'] == data_kind['P1M'][1]) & (df['Z_model'] == data_kind['P1M'][3]) ) | ( (df['W_model'] == data_kind['P2M'][1]) & (df['Z_model'] == data_kind['P2M'][3]) ) ) &
            ( ( (df['W_category'] == data_kind['P1M'][0]) & (df['Z_category'] == data_kind['P1M'][2]) ) | ( (df['W_category'] == data_kind['P2M'][0]) & (df['Z_category'] == data_kind['P2M'][2]) ) )
            )
    data_subset = df[mask]

    for method in ['proximal_P1M', 'proximal_P2M']:
        method_subset = data_subset.loc[data_subset["method"] == method]

        final_string += f'\\begin{{table*}}[ht]\n'
        final_string += f'\t\\centering\n'
        final_string += f'\t\\resizebox{{0.9\\linewidth}}{{!}}{{\n'
        final_string += f'\t\t\\begin{{tabular}}{{l l l r}}\n'
        final_string += f'\t\t\t\\toprule\n'
        final_string += f'\t\t\tOracle? & Proxies & Metric & $U$={method_subset['oracle'].iloc[0]} \\\\ [0.5ex]\n'
        final_string += f'\t\t\t\\toprule\n'
        final_string += f'\t\t\tYes & -- & 1 - $p(U=1)$ & {(1-method_subset['oracle_positivity'].iloc[0]):.3f} \\\\ \n'
        final_string += f'\t\t\t\midrule\\\n'
        final_string += f'\t\t\t\\multirow{{5}}{{*}}{{Yes}} &\n'
        final_string += f'\t\t\t\multirow{{5}}{{*}}{{$W$ from {method_subset['W_model'].iloc[0].capitalize()} on ${{\\bf T}}^{{\\text{{pre}}}}_1$ ({method_subset['W_category'].iloc[0]})}} & $\\gamma_{{WU.{{\\bf C}}}}$& {method_subset['gamma WU.C'].iloc[0]:.3f} \\\\ \n'
        final_string += f'\t\t\t&& Accuracy & {method_subset['W_accuracy'].iloc[0]:.3f} \\\\\n'
        final_string += f'\t\t\t&& $p(W=1)$ & {method_subset['p(W=1)'].iloc[0]:.3f} \\\\\n'
        final_string += f'\t\t\t&& Precision & {method_subset['W precision'].iloc[0]:.3f} \\\\\n'
        final_string += f'\t\t\t&& Recall & {method_subset['W recall'].iloc[0]:.3f} \\\\\n'
        final_string += f'\t\t\t\\midrule\n'
        final_string += f'\t\t\t\\multirow{{5}}{{*}}{{Yes}} &\n'
        final_string += f'\t\t\t\multirow{{5}}{{*}}{{$Z$ from {method_subset['Z_model'].iloc[0].capitalize()} on ${{\\bf T}}^{{\\text{{pre}}}}_1$ ({method_subset['Z_category'].iloc[0]})}} & $\\gamma_{{ZU.{{\\bf C}}}}$& {method_subset['gamma ZU.C'].iloc[0]:.3f} \\\\ \n'
        final_string += f'\t\t\t&& Accuracy & {method_subset['Z_accuracy'].iloc[0]:.3f} \\\\\n'
        final_string += f'\t\t\t&& $p(Z=1)$ & {method_subset['p(Z=1)'].iloc[0]:.3f} \\\\\n'
        final_string += f'\t\t\t&& Precision & {method_subset['Z precision'].iloc[0]:.3f} \\\\\n'
        final_string += f'\t\t\t&& Recall & {method_subset['Z recall'].iloc[0]:.3f} \\\\\n'
        final_string += f'\t\t\t\\midrule\n'
        final_string += f'\t\t\tNo & $Z, W$ & Raw Agreement Rate; $p(W=Z)$ & {method_subset['WZ agreement'].iloc[0]:.3f} \\\\ \n'
        final_string += f'\t\t\t\\bottomrule\n'
        final_string += f'\t\t\\end{{tabular}}}}\n'
        final_string += f'\t\\caption{{Key metrics for the diagnosis {LONG_NAMES[method_subset['oracle'].iloc[0]]} when inferring proxies from {method_subset['W_category'].iloc[0]} and {method_subset['Z_category'].iloc[0]} clinicians\' notes with {METHOD_COUNT[method_subset['method'].iloc[0]]}.}}\n'
        final_string += f'\t\\label{{tab:semi_synthetic_{method_subset['oracle'].iloc[0]}_{method_subset['W_category'].iloc[0]}_{method_subset['Z_category'].iloc[0]}_{method}}}\n'
        final_string += f'\\end{{table*}}\n'

        final_string += f'\n'

    print(final_string)