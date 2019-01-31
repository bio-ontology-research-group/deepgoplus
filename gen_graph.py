#!/usr/bin/env python
import click as ck
import gzip

@ck.command()
@ck.option(
    '--string-db-file', '-sdb', default='data/string/protein.links.v10.5.txt.gz',
    help='StringDB protein links file')
@ck.option(
    '--orth-file', '-of', default='data/string/COG.mappings.v10.5.txt.gz',
    help='StringDB orthology mappings')
@ck.option(
    '--out-file', '-o',
    default='data/string/graph.out',
    help='Graph for Word2Vec')
@ck.option(
    '--mapping-file', '-mf',
    default='data/string/graph_mapping.out',
    help='Protein ID to graph id mapping')
def main(string_db_file, orth_file, out_file, mapping_file):
    mapping = dict()
    rf = open(out_file, 'w')

    # Reading interactions data
    inter = 'interaction'
    orth = 'ortholog'
    mapping[inter] = len(mapping)
    mapping[orth] = len(mapping)
            
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
                mapping[p1] = len(mapping)
            if p2 not in mapping:
                mapping[p2] = len(mapping)
            id1 = str(mapping[p1])
            id2 = str(mapping[p2])
            id3 = str(mapping[inter])
            rf.write(id1 + ' ' + id2 + ' ' + id3 + '\n')
            rf.write(id2 + ' ' + id1 + ' ' + id3 + '\n')

    orthologs = {}
    with gzip.open(orth_file, 'rt') as f:
        next(f)
        for line in f:
            it = line.strip().split('\t')
            o_id = it[3]
            p_id = it[0]
            if o_id not in orthologs:
                orthologs[o_id] = []
            orthologs[o_id].append(p_id)

    for g_id, prots in orthologs.items():
        for i in range(len(prots)):
            for j in range(i + 1, len(prots)):
                if prots[i] not in mapping:
                    mapping[prots[i]] = len(mapping)
                if prots[j] not in mapping:
                    mapping[prots[j]] = len(mapping)
                id1 = str(mapping[prots[i]])
                id2 = str(mapping[prots[j]])
                id3 = str(mapping[orth])
                rf.write(id1 + ' ' + id2 + ' ' + id3 + '\n')
                rf.write(id2 + ' ' + id1 + ' ' + id3 + '\n')

    rf.close()

    with open(mapping_file, 'w') as f:
        for key, value in mapping.items():
            f.write(key + '\t' + str(value) + '\n')



if __name__ == '__main__':
    main()
