for i in 19 43 48 49 59 63 64 69 74
do
	python deepgoplus.py -ld -bs 32 -d gpu:1 -pi $i >> job_${i}.out 2>&1
done
