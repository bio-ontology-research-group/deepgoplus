import numpy as np

AALETTER = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
AANUM = len(AALETTER)
AAINDEX = dict()
for i in range(len(AALETTER)):
    AAINDEX[AALETTER[i]] = i + 1
INVALID_ACIDS = set(['U', 'O', 'B', 'Z', 'J', 'X', '*'])
MAXLEN = 2000
NGRAMS = {}
for i in range(20):
    for j in range(20):
        for k in range(20):
            ngram = AALETTER[i] + AALETTER[j] + AALETTER[k]
            index = 400 * i + 20 * j + k + 1
            NGRAMS[ngram] = index

def is_ok(seq):
    for c in seq:
        if c in INVALID_ACIDS:
            return False
    return True

def to_ngrams(seq):
    l = min(MAXLEN, len(seq) - 3)
    ngrams = np.zeros((l,), dtype=np.int32)
    for i in range(l):
        ngrams[i] = NGRAMS.get(seq[i: i + 3], 0)
    return ngrams

def to_onehot(seq, start=0):
    onehot = np.zeros((MAXLEN, 21), dtype=np.int32)
    l = min(MAXLEN, len(seq))
    for i in range(start, start + l):
        onehot[i, AAINDEX.get(seq[i - start], 0)] = 1
    onehot[0:start, 0] = 1
    onehot[start + l:, 0] = 1
    return onehot
