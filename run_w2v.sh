word2vec -train data/string/walks.out -output data/string/vectors.out -size 256 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 50
