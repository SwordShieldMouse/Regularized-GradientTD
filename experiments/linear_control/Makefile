all:
	parallel --jobs 2 python3.6 src/main.py 10 ./experiments/MC-TC/PFGQ.json {1} ::: 0 1 2
	parallel --jobs 2 python3.6 src/main.py 10 ./experiments/MC-TC/PFGQUntrunc.json {1} ::: 0 1 2
	parallel --jobs 2 python3.6 src/main.py 10 ./experiments/MC-TC/PFGQScaledGrad.json {1} ::: 0 1 2
	parallel --jobs 2 python3.6 src/main.py 10 ./experiments/MC-TC/QC.json {1} ::: 0 1 2 3
	parallel --jobs 2 python3.6 src/main.py 10 ./experiments/MC-TC/QRC.json {1} ::: 0 1 2 3
	parallel --jobs 2 python3.6 src/main.py 10 ./experiments/MC-TC/QLearning.json {1} ::: 0 1 2 3
	python3.6 analysis/learning_curve.py ./experiments/MC-TC/*
lc:
	python3.6 analysis/learning_curve.py ./experiments/MC-TC/*
