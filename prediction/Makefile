all:
	parallel --jobs 2 python3.6 src/main.py 10 ./experiments/online/Baird/best/{1}.json 0 ::: pfgtd gtd2 htd tdc td tdrc vtrace
plot:
	python3.6 analysis/learning_curve.py ./experiments/online/Baird/best/{gtd2,tdc,htd,tdrc,pfgtd}.json
