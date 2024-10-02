"""
Oct. 2, 2024

This file contains code for creating latex code that creates
tables for data providing evidence for text independence.
"""
import pickle

def create_table_code(filename):
    final_string = ''

    table_data = pickle.load(open(filename, "rb"))

    for pair in table_data:
        data = table_data[pair]

        final_string += f'\\begin{{table*}}[h!]\n'
        final_string += f'\\centering\n'
        final_string += f'\\resizebox{{0.63\\linewidth}}{{!}}{{\n'
        final_string += f'\\begin{{tabular}}{{cccccc}}\n'
        final_string += f'\\toprule\n'
        final_string += f' & {data["combined_text_words"][0]} & {data["combined_text_words"][1]} & {data["combined_text_words"][2]} & {data["combined_text_words"][3]} & {data["combined_text_words"][4]} \\\\ \\midrule\n'
        final_string += f'{pair[0]} Notes       & {data["T_1_column_averages"][0]:.3f}     & {data["T_1_column_averages"][1]:.3f}     & {data["T_1_column_averages"][2]:.3f}     & {data["T_1_column_averages"][3]:.3f}     & {data["T_1_column_averages"][4]:.3f}     \\\\\n'
        final_string += f'{pair[1]} Notes       & {data["T_2_column_averages"][0]:.3f}     & {data["T_2_column_averages"][1]:.3f}     & {data["T_2_column_averages"][2]:.3f}     & {data["T_2_column_averages"][3]:.3f}     & {data["T_2_column_averages"][4]:.3f}     \\\\ \\bottomrule\n'
        final_string += f'\\end{{tabular}}}}\n'
        final_string += f'\\caption{{Positivity rates in {pair[0]} and {pair[1]} notes of the 5 most important features when fitting the Tfidf Vectorizer on the concatenation of the two note categories.}}\n'
        final_string += f'\\label{{tab:combined_vocab_{pair[0]}_{pair[1]}}}\n'
        final_string += f'\\end{{table*}}\n'

        final_string += f'\n'

        final_string += f'\\begin{{table*}}[h!]\n'
        final_string += f'\\centering\n'
        final_string += f'\\resizebox{{0.53\\linewidth}}{{!}}{{\n'
        final_string += f'\\begin{{tabular}}{{cc}}\n'
        final_string += f'\\toprule\n'
        final_string += f' & 5 Most Important Features             \\\\ \\midrule\n'
        final_string += f'{pair[0]} Notes     & {{[}}{data["T_1_words"][0]}, {data["T_1_words"][1]}, {data["T_1_words"][2]}, {data["T_1_words"][3]}, {data["T_1_words"][4]}{{]}} \\\\\n'
        final_string += f'{pair[1]} Notes     & {{[}}{data["T_2_words"][0]}, {data["T_2_words"][1]}, {data["T_2_words"][2]}, {data["T_2_words"][3]}, {data["T_2_words"][4]}{{]}} \\\\ \\bottomrule\n'
        final_string += f'\\end{{tabular}}}}\n'
        final_string += f'\\caption{{The 5 most important features of {pair[0]} and {pair[1]} notes for patients who have notes on both categories and when each note is fitted on a unique Tfidf Vectorizer.}}\n'
        final_string += f'\\label{{tab:separate_vocab_{pair[0]}_{pair[1]}}}\n'
        final_string += f'\\end{{table*}}\n'

        final_string += f'\n'

    return final_string