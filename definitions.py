#!/usr/bin/env python
import click as ck
import numpy as np
import pandas as pd

@ck.command()
def main():
    defs = {}
    with open('data/go_corpus.txt') as f:
        for line in f:
            it = line.strip().split(': ')
            defs[it[0]] = it[1].split(' and ')
    global gdefs
    gdefs = {}
    expand_definitions(defs)

def get_definition(defs, go_id, depth=0):
    if go_id in gdefs:
        return gdefs[go_id]
    dn = defs[go_id]
    new_def = []
    for g_id in dn:
        if g_id in defs:
            new_def +=  get_definition(defs, g_id, depth + 1)
        else:
            new_def.append(g_id)
    gdefs[go_id] = new_def
    return new_def

def expand_definitions(defs):
    new_defs = {}
    for go_id, val in defs.items():
        new_val = []
        for i, it in enumerate(val):
            if it in defs:
                new_val += get_definition(defs, it)
            else:
                new_val.append(it)
            new_defs[go_id] = new_val

    f = open('data/go_corpus_expanded.txt', 'w')
    for go_id, val in new_defs.items():
        f.write(go_id + ': ' + ' and '.join(val) + '\n')
    f.close()
    
def save_nodefs(defs):
    nodefs = set()
    cnt_defs = 0
    for key, val in defs.items():
        for it in val:
            it = it.split(' some ')
            x = it[0] if len(it) == 1 else it[1]
            if x not in defs:
                nodefs.add(x)
            else:
                print(x, 'has definition')
                cnt_defs += 1
    cnts = {}
    for it in nodefs:
        s = it.split('_')
        if s[0] not in cnts:
            cnts[s[0]] = list()
        cnts[s[0]].append(it)
    for key, val in cnts.items():
        f = open('data/nodef_' + key + '.txt', 'w')
        for it in val:
            f.write(it + '\n')
        f.close()
            
if __name__ == '__main__':
    main()
