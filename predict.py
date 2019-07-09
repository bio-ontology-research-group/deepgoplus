#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from subprocess import Popen, PIPE
import time
from utils import Ontology
from aminoacids import to_onehot

MAXLEN = 2000

@ck.command()
@ck.option('--in-file', '-if', help='Input FASTA file', required=True)
@ck.option('--out-file', '-of', default='results.tsv', help='Output result file')
@ck.option('--go-file', '-gf', default='data/go.obo', help='Gene Ontology')
@ck.option('--model-file', '-mf', default='data/model.h5', help='Tensorflow model file')
@ck.option('--terms-file', '-tf', default='data/terms.pkl', help='List of predicted terms')
@ck.option('--annotations-file', '-tf', default='data/train_data.pkl', help='Experimental annotations')
@ck.option('--chunk-size', '-cs', default=1000, help='Number of sequences to read at a time')
@ck.option('--diamond-file', '-df', default='data/test_diamond.res', help='Diamond Mapping file')
@ck.option('--threshold', '-t', default=0.0, help='Prediction threshold')
@ck.option('--batch-size', '-bs', default=32, help='Batch size for prediction model')
@ck.option('--alpha', '-a', default=0.5, help='Alpha weight parameter')
def main(in_file, out_file, go_file, model_file, terms_file, annotations_file,
         chunk_size, diamond_file, threshold, batch_size, alpha):
    # Load GO and read list of all terms
    go = Ontology(go_file, with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['terms'].values.flatten()

    # Read known experimental annotations
    annotations = {}
    df = pd.read_pickle(annotations_file)
    for row in df.itertuples():
        annotations[row.proteins] = set(row.annotations)

    diamond_preds = {}
    mapping = {}
    with open(diamond_file, 'r') as f:
        for line in f:
            it = line.strip().split()
            if it[0] not in mapping:
                mapping[it[0]] = {}
            mapping[it[0]][it[1]] = float(it[2])
    for prot_id, sim_prots in mapping.items():
        annots = {}
        allgos = set()
        total_score = 0.0
        for p_id, score in sim_prots.items():
            allgos |= annotations[p_id]
            total_score += score
        allgos = list(sorted(allgos))
        sim = np.zeros(len(allgos), dtype=np.float32)
        for j, go_id in enumerate(allgos):
            s = 0.0
            for p_id, score in sim_prots.items():
                if go_id in annotations[p_id]:
                    s += score
            sim[j] = s / total_score
        for go_id, score in zip(allgos, sim):
            annots[go_id] = score
        diamond_preds[prot_id] = annots
    
    # Load CNN model
    model = load_model(model_file)

    start_time = time.time()
    total_seq = 0
    w = open(out_file, 'w')
    for prot_ids, sequences in read_fasta(in_file, chunk_size):
        total_seq += len(prot_ids)
        deep_preds = {}
        ids, data = get_data(sequences)

        preds = model.predict(data, batch_size=batch_size)
        assert preds.shape[1] == len(terms)
        for i, j in enumerate(ids):
            prot_id = prot_ids[j]
            if prot_id not in deep_preds:
                deep_preds[prot_id] = {}
            for l in range(len(terms)):
                if preds[i, l] >= 0.01: # Filter out very low scores
                    if terms[l] not in deep_preds[prot_id]:
                        deep_preds[prot_id][terms[l]] = preds[i, l]
                    else:
                        deep_preds[prot_id][terms[l]] = max(
                            deep_preds[prot_id][terms[l]], preds[i, l])
        # Combine diamond preds and deepgo
        for prot_id in prot_ids:
            annots = {}
            if prot_id in diamond_preds:
                for go_id, score in diamond_preds[prot_id].items():
                    annots[go_id] = score * alpha
            for go_id, score in deep_preds[prot_id].items():
                if go_id in annots:
                    annots[go_id] += (1 - alpha) * score
                else:
                    annots[go_id] = (1 - alpha) * score
            # Propagate scores with ontology structure
            gos = list(annots.keys())
            for go_id in gos:
                for g_id in go.get_anchestors(go_id):
                    if g_id in annots:
                        annots[g_id] = max(annots[g_id], annots[go_id])
                    else:
                        annots[g_id] = annots[go_id]
                
            w.write(prot_id)
            for go_id, score in annots.items():
                if score >= threshold:
                    w.write('\t' + go_id + '|%.3f' % score)
            w.write('\n')
    w.close()
    total_time = time.time() - start_time
    print('Total prediction time for %d sequences is %d' % (total_seq, total_time))


def read_fasta(filename, chunk_size):
    seqs = list()
    info = list()
    seq = ''
    inf = ''
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq != '':
                    seqs.append(seq)
                    info.append(inf)
                    if len(info) == chunk_size:
                        yield (info, seqs)
                        seqs = list()
                        info = list()
                    seq = ''
                inf = line[1:].split()[0]
            else:
                seq += line
        seqs.append(seq)
        info.append(inf)
    yield (info, seqs)

def get_data(sequences):
    pred_seqs = []
    ids = []
    for i, seq in enumerate(sequences):
        if len(seq) > MAXLEN:
            st = 0
            while st < len(seq):
                pred_seqs.append(seq[st: st + MAXLEN])
                ids.append(i)
                st += MAXLEN - 128
        else:
            pred_seqs.append(seq)
            ids.append(i)
    n = len(pred_seqs)
    data = np.zeros((n, MAXLEN, 21), dtype=np.float32)
    
    for i in range(n):
        seq = pred_seqs[i]
        data[i, :, :] = to_onehot(seq)
    return ids, data


if __name__ == '__main__':
    main()
