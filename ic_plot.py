from matplotlib import pyplot as plt
import click as ck
import numpy as np

@ck.command()
def main():
    ics = list()
    cnt = 0
    with open('data-cafa/ic_dist.txt') as f:
        for line in f:
            it = line.strip().split()
            ic = float(it[1])
            cnt = max(ic, cnt)
            ics.append(float(it[1]))
    print(cnt)
    return
    plt.hist(ics, color='blue', bins=100)
    plt.title('Distribution of information content values')
    plt.xlabel('Information Content')
    plt.ylabel('Number of classes')
    plt.savefig('ic-hist.pdf')
    plt.show()


if __name__ == '__main__':
    main()
