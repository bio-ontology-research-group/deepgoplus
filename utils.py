from collections import deque
import warnings
import pandas as pd
import numpy as np
from xml.etree import ElementTree as ET

BIOLOGICAL_PROCESS = 'GO:0008150'
MOLECULAR_FUNCTION = 'GO:0003674'
CELLULAR_COMPONENT = 'GO:0005575'
FUNC_DICT = {
    'cc': CELLULAR_COMPONENT,
    'mf': MOLECULAR_FUNCTION,
    'bp': BIOLOGICAL_PROCESS}

EXP_CODES = set(['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC'])


def is_exp_code(code):
    return code in EXP_CODES


class GeneOntology(object):

    def __init__(self, filename='data/go.obo', with_rels=False):
        self.go = self.load(filename, with_rels)

    def has_term(self, go_id):
        return go_id in self.go

    def load(self, filename, with_rels):
        # Reading Gene Ontology from OBO Formatted file
        go = dict()
        obj = None
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == '[Term]':
                    if obj is not None:
                        go[obj['id']] = obj
                    obj = dict()
                    obj['is_a'] = list()
                    obj['part_of'] = list()
                    obj['regulates'] = list()
                    obj['alt_ids'] = list()
                    obj['is_obsolete'] = False
                    continue
                elif line == '[Typedef]':
                    obj = None
                else:
                    if obj is None:
                        continue
                    l = line.split(": ")
                    if l[0] == 'id':
                        obj['id'] = l[1]
                    elif l[0] == 'alt_id':
                        obj['alt_ids'].append(l[1])
                    elif l[0] == 'is_a':
                        obj['is_a'].append(l[1].split(' ! ')[0])
                    elif with_rels and l[0] == 'relationship':
                        it = l[1].split()
                        # add all types of relationships
                        obj['is_a'].append(it[1])
                    elif l[0] == 'name':
                        obj['name'] = l[1]
                    elif l[0] == 'is_obsolete' and l[1] == 'true':
                        obj['is_obsolete'] = True
        if obj is not None:
            go[obj['id']] = obj
        for go_id in list(go.keys()):
            for g_id in go[go_id]['alt_ids']:
                go[g_id] = go[go_id]
            if go[go_id]['is_obsolete']:
                del go[go_id]
        for go_id, val in go.items():
            if 'children' not in val:
                val['children'] = set()
            for p_id in val['is_a']:
                if p_id in go:
                    if 'children' not in go[p_id]:
                        go[p_id]['children'] = set()
                    go[p_id]['children'].add(go_id)
        return go


    def get_anchestors(self, go_id):
        if go_id not in self.go:
            return set()
        go_set = set()
        q = deque()
        q.append(go_id)
        while(len(q) > 0):
            g_id = q.popleft()
            if g_id not in go_set:
                go_set.add(g_id)
                for parent_id in self.go[g_id]['is_a']:
                    if parent_id in self.go:
                        q.append(parent_id)
        return go_set


    def get_parents(self, go_id):
        if go_id not in self.go:
            return set()
        go_set = set()
        for parent_id in self.go[go_id]['is_a']:
            if parent_id in self.go:
                go_set.add(parent_id)
        return go_set


    def get_go_set(self, go_id):
        if go_id not in self.go:
            return set()
        go_set = set()
        q = deque()
        q.append(go_id)
        while len(q) > 0:
            g_id = q.popleft()
            if g_id not in go_set:
                go_set.add(g_id)
                for ch_id in self.go[g_id]['children']:
                    q.append(ch_id)
        return go_set

def read_fasta(lines):
    seqs = list()
    info = list()
    seq = ''
    inf = ''
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            if seq != '':
                seqs.append(seq)
                info.append(inf)
                seq = ''
            inf = line[1:]
        else:
            seq += line
    seqs.append(seq)
    info.append(inf)
    return info, seqs


class DataGenerator(object):

    def __init__(self, batch_size, is_sparse=False):
        self.batch_size = batch_size
        self.is_sparse = is_sparse

    def fit(self, inputs, targets=None):
        self.start = 0
        self.inputs = inputs
        self.targets = targets
        if isinstance(self.inputs, tuple) or isinstance(self.inputs, list):
            self.size = self.inputs[0].shape[0]
        else:
            self.size = self.inputs.shape[0]
        self.has_targets = targets is not None

    def __next__(self):
        return self.next()

    def reset(self):
        self.start = 0

    def next(self):
        if self.start < self.size:
            batch_index = np.arange(
                self.start, min(self.size, self.start + self.batch_size))
            if isinstance(self.inputs, tuple) or isinstance(self.inputs, list):
                res_inputs = []
                for inp in self.inputs:
                    if self.is_sparse:
                        res_inputs.append(
                            inp[batch_index, :].toarray())
                    else:
                        res_inputs.append(inp[batch_index, :])
            else:
                if self.is_sparse:
                    res_inputs = self.inputs[batch_index, :].toarray()
                else:
                    res_inputs = self.inputs[batch_index, :]
            self.start += self.batch_size
            if self.has_targets:
                if self.is_sparse:
                    labels = self.targets[batch_index, :].toarray()
                else:
                    labels = self.targets[batch_index, :]
                return (res_inputs, labels)
            return res_inputs
        else:
            self.reset()
            return self.next()

