#!/usr/bin/env python
import click as ck
import gzip

@ck.command()
@ck.option(
    '--string-db-file', '-sdb', default='data/string/protein.links.v10.5.txt.gz',
    help='StringDB protein links file')
@ck.option(
    '--out-file', '-o',
    default='data/string/graph.out',
    help='Graph for Word2Vec')
@ck.option(
    '--mapping-file', '-mf',
    default='data/string/graph_mapping.out',
    help='Protein ID to graph id mapping')
def main(string_db_file, out_file, mapping_file):
    mapping = dict()
    size = 0
    rf = open(out_file, 'w')

    # Reading interactions data
    edge = 0
    inter = 'interaction'
    mapping[inter] = size
    size += 1
            
    with gzip.open(string_db_file, 'rt') as f:
        next(f)
        for line in f:
            items = line.strip().split(' ')
            p1 = items[0]
            p2 = items[1]
            score = int(items[2])
            if score < 300:
                continue
            if p1 not in mapping:
                mapping[p1] = size
                size += 1
            if p2 not in mapping:
                mapping[p2] = size
                size += 1
            id1 = str(mapping[p1])
            id2 = str(mapping[p2])
            id3 = str(mapping[inter])
            edge += 1
            rf.write(id1 + ' ' + id2 + ' ' + id3 + '\n')
            rf.write(id2 + ' ' + id1 + ' ' + id3 + '\n')

    print('Nodes', (len(mapping) - edge_types))

    rf.close()

    with open(mapping_file, 'w') as f:
        for key, value in mapping.iteritems():
            f.write(key + '\t' + str(value) + '\n')



if __name__ == '__main__':
    main()
