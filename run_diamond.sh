diamond blastp -q data/swissprot_exp.fa -o data/diamond_mapping.out -e 1\
	-d data/string/stringdb --max-target-seqs 1 --outfmt 6 qseqid sseqid
