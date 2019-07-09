from matplotlib import pyplot as plt
import click as ck
import numpy as np

@ck.command()
def main():
    groups, performance, cs, colors, yerr = get_cc_data()
    cs = map(lambda x: 'C={}'.format(x), cs)
    n = len(groups)
    y_pos = np.arange(n)
    fig, ax = plt.subplots()
    plt.rc('text', usetex=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('none')
    ax.tick_params(axis='both', which='major', labelsize='x-large')
    ax.bar(
        y_pos, performance, align='center', color=colors,
        edgecolor="none")
    x = np.arange(-1, n, 0.2)
    y = [performance[-1]] * len(x)
    ax.plot(x, y, 's', markersize=3, color='#00539f', markeredgecolor="none")
    y = [performance[-2]] * len(x)
    ax.plot(x, y, 's', markersize=3, color='#c4302b', markeredgecolor="none")
    tcolors = ['black'] * n
    tcolors[-1] = tcolors[-2] = 'white'
    for xloc, c in zip(y_pos, cs):
        ax.text(xloc, 0.05, c, clip_on=True, rotation='vertical',
                va='bottom', ha='center', color=tcolors[xloc], size='x-large')
    plt.xticks(y_pos, groups, rotation=45, ha="right", size='x-large')
    plt.xlim([-1, n])
    plt.ylabel(r'$F_{\max}$', size='x-large')
    plt.title(r'\textbf{Cellular Component}')
    plt.tight_layout()
    plt.savefig('cc.pdf')
    plt.show()



def get_data():
    groups = ['Zhu Lab', 'DeepGOPlus', 'orengo-funfams', 'Tian Lab', 'Kihara Lab', 'INGA-Tosatto', 'schoofcropbiobonn', 'Argot25Toppo Lab', 'Temple', 'TurkuBioNLP1', 'Holm', 'Naive', 'BLAST']
    performance = [0.62, 0.56, 0.54, 0.53, 0.53, 0.52, 0.52, 0.52, 0.52, 0.52, 0.51, 0.33, 0.42]
    cs = [1.00, 1.00, 0.85, 0.92, 1.00, 0.93, 1.00, 0.99, 0.98, 0.79, 0.88, 0.93, 0.93]
    colors = ['#999999'] * len(groups)
    colors[1] = '#42aaff'
    colors[-1] = '#00539f'
    colors[-2] = '#c4302b'
    yerr = [0.02] * len(groups)
    return groups, performance, cs, colors, yerr

def get_bp_data():
    groups = ['Zhu Lab', 'DeepGOPlus', 'INGA-Tosatto', 'Argot25Toppo Lab', 'orengo-funfams', 'Zhang-Freddolino Lab', 'Tian Lab', 'Kihara Lab', 'Holm', 'Temple', 'TurkuBioNLP1', 'Naive', 'BLAST']
    performance = [0.40, 0.39, 0.39, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.37, 0.37, 0.255, 0.263]
    cs = [0.98, 1.00, 0.99, 0.98, 0.87, 1.00, 0.94, 1.00, 0.88, 0.99, 0.96, 0.97, 0.97]
    colors = ['#999999'] * len(groups)
    colors[1] = '#42aaff'
    colors[-1] = '#00539f'
    colors[-2] = '#c4302b'
    yerr = [0.02] * len(groups)
    return groups, performance, cs, colors, yerr

def get_cc_data():
    groups = ['DeepGOPlus', 'Zhu Lab', 'Kihara Lab', 'INGA-Tosatto', 'Zhang-Freddolino Lab', 'TurkuBioNLP1', 'DeepMaster', 'Jones-UCL', 'Argot25Toppo Lab', 'orengo-funfams', 'NCCUCS',  'Naive', 'BLAST']
    performance = [0.61, 0.61, 0.61, 0.60, 0.60, 0.60, 0.60, 0.59, 0.59, 0.59, 0.58, 0.54, 0.46]
    cs = [1.00, 1.00, 1.00, 0.99, 1.00, 0.96, 1.00, 1.00, 1.00, 0.87, 1.00, 0.97, 0.97]
    colors = ['#999999'] * len(groups)
    colors[0] = '#42aaff'
    colors[-1] = '#00539f'
    colors[-2] = '#c4302b'
    yerr = [0.02] * len(groups)
    return groups, performance, cs, colors, yerr

if __name__ == '__main__':
    main()
