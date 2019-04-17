# Run diamond to get similar sequences
diamond blastp -d data/train_data.dmnd -q $1 --outfmt 6 qseqid sseqid bitscore > data/diamond.res

# Run prediction
python predict.py -if $1 -of $2 -df data/diamond.res
