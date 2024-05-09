import pickle

def create_table(filename):
    save = pickle.load(open(filename, 'rb'))

    CI_COLOR = {
        'P1M': 'blue',
        'P1M, same': 'red',
        'P2M': 'blue',
        'P2M, same': 'red'
    }

    CI_CHECKMARK = {
        'P1M': f'^\\checkmark',
        'P1M, same': '',
        'P2M': f'^\\checkmark',
        'P2M, same': ''
    }

    final_string = ''

    for data in save:
        final_string += f'\t\t{data['pipeline']} & {{\\color{{{CI_COLOR[data['pipeline']]}}} $({data['or_ci_low']:.3g}, {data['or_ci_high']:.3g}){CI_CHECKMARK[data['pipeline']]}$}} & ${data['est_ace']:.3f}$ & ${data['bias']:.3f}$ & $({data['ace_ci'][0]:.3f}, {data['ace_ci'][1]:.3f})$ & {data['ci_cov']} \\\\\n'

    return final_string