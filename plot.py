from matplotlib import pyplot as plt
import click as ck
import numpy as np

@ck.command()
def main():
    groups, performance, cs, colors = get_data()
    y_pos = np.arange(len(groups))
    yerr = [0.02] * len(groups)
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
    return groups, performance, cs, colors

if __name__ == '__main__':
    main()
