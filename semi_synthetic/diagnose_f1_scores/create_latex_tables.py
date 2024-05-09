import pickle
import pandas as pd

SHORT_NAME_DICT = {
    'Hypertension NOS': 'Hypertension',
    'Crnry athrscl natve vssl': 'Coronary atherosclerosis',
    'Atrial fibrillation': 'Atrial fibrillation',
    'CHF NOS': 'Congestive heart failure',
    'DMII wo cmp nt st uncntr': 'Diabetes',
    'Hyperlipidemia NEC/NOS': 'Hyperlipidemia',
    'Acute kidney failure NOS': 'Acute kidney failure',
    'Need prphyl vc vrl hepat': 'Need herpes vaccination',
    'Acute respiratry failure': 'Acute respiratory failure'
}

def sectionF(filename):
    """
    Create strings for tables in Section F of the appendix that show F1 scores of
    different supervised classifiers if the F1 score is above the min_threshold.
    """
    # all_results is a dictionary of dictionaries
    all_results = pickle.load(open(filename, 'rb'))

    # results_to_sort is a list of dictionaries
    results_to_sort = []

    # convert the dictionary of dictionaries into a list of dictionaries with only
    # the metrics that we want to keep
    for oracle, note_category in all_results:
        result = all_results[(oracle, note_category)]
        
        results_to_sort.append({'U': SHORT_NAME_DICT[result['U']], 'note_category': result['text'], 'accuracy': result['accuracy'],
                                'precision': result['precision'], 'recall': result['recall'], 'f1': result['f1'], 'p(U=1)': result['p(U)=1']})
        
    results_sorted = sorted(results_to_sort, key=lambda ele: ele['f1'], reverse=True)

    final_string = ''

    for result in results_sorted:
        final_string += f'{result['U']} & {result['note_category']} & {result['p(U=1)']:.3f} & {result['f1']:.3f} & {result['accuracy']:.3f} & {result['precision']:.3f} & {result['recall']:.3f} \\\\ \n'

    print(final_string)