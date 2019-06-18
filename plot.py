from matplotlib import pyplot as plt
import click as ck
import numpy as np

@ck.command()
def main():
    groups, performance, cs, colors = get_data()
    y_pos = np.arange(len(groups))
    plt.bar(y_pos, performance, yerr=yerr, align='center', alpha=0.5, color=colors)
    plt.xticks(y_pos, groups)
    plt.ylabel('Fmax')
    plt.title('Molecular Function')

    plt.show()



def get_data():
    groups = ['Zhu Lab', 'orengo-funfams', 'Tian Lab', 'Kihara Lab', 'INGA-Tosatto', 'schoofcropbiobonn', 'Argot25Toppo Lab', 'Temple', 'TurkuBioNLP1', 'Holm', 'Naive', 'BLAST']
    performance = [0.62, 0.54, 0.53, 0.53, 0.52, 0.52, 0.52, 0.52, 0.52, 0.51, 0.33, 0.42]
    cs = [1.00, 0.85, 0.92, 1.00, 0.93, 1.00, 0.99, 0.98, 0.79, 0.88, 0.93, 0.93]
    colors = ['grey'] * len(groups)
    colors[-1] = 'blue'
    colors[-2] = 'red'
    yerr = [0.02] * len(groups)
    return groups, performance, cs, colors, yerr

def get_bp_data():
    groups = ['Zhu Lab', 'INGA-Tosatto', 'Argot25Toppo Lab', 'orengo-funfams', 'Zhang-Freddolino Lab', 'Tian Lab', 'Kihara Lab', 'Holm', 'Temple', 'TurkuBioNLP1', 'Naive', 'BLAST']
    performance = [0.40, 0.39, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.37, 0.37, 0.26, 0.26]
    cs = [0.98, 0.99, 0.98, 0.87, 1.00, 0.94, 1.00, 0.88, 0.99, 0.96, 0.97, 0.97]
    colors = ['grey'] * len(groups)
    colors[-1] = 'blue'
    colors[-2] = 'red'
    yerr = [0.02] * len(groups)
    return groups, performance, cs, colors, yerr

def get_cc_data():
    groups = ['Zhu Lab', 'Kihara Lab', 'INGA-Tosatto', 'Zhang-Freddolino Lab', 'TurkuBioNLP1', 'DeepMaster', 'Jones-UCL', 'Argot25Toppo Lab', 'orengo-funfams', 'NCCUCS',  'Naive', 'BLAST']
    performance = [0.62, 0.54, 0.53, 0.53, 0.52, 0.52, 0.52, 0.52, 0.52, 0.51, 0.33, 0.42]
    cs = [1.00, 0.85, 0.92, 1.00, 0.93, 1.00, 0.99, 0.98, 0.79, 0.88, 0.93, 0.93]
    colors = ['grey'] * len(groups)
    colors[-1] = 'blue'
    colors[-2] = 'red'
    yerr = [0.02] * len(groups)
    return groups, performance, cs, colors, yerr

if __name__ == '__main__':
    main()
