for i in $(seq 600 1 1199)
do
	python deepgoplus.py -d gpu:1 -pi $i > job_${i}.out 2>&1
done
