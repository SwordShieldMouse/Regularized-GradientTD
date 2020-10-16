NUM_RUNS = 200

best:
	$(Run) $(BestOnline); $(RunBatch) $(bestBatch); $(PlotBest); $(PlotBestBatch)

# =========================================
# --- BEST SETTINGS FOR EACH EXPERIMENT ---
# =========================================

# --- [[ ONLINE ]] ---
bestOnline:
	$(Run) $(BestOnline); $(PlotBest)
bestBaird:
	$(Run) $(BestBaird); $(plotBestBaird)
bestBoyan:
	$(Run) $(BestBoyan); $(PlotBestBoyan)
bestRW:
	$(Run) $(BestRandomWalk); $(PlotBestRW)

# --- PLOT RESULTS ---
plotBestOnline:
	$(PlotBestOnline)
plotBestBaird:
	$(PlotBestBaird)
plotBestBoyan:
	$(PlotBestBoyan)
plotBestRW:
	$(PlotBestRW)

# --- [[ BATCH ]] ---
bestBatch:
	$(RunBatch) $(BestBatch); $(PlotBestBatch)
bestBatchBaird:
	$(RunBatch) $(BestBatchBaird); $(plotBestBatchBaird)
bestBatchBoyan:
	$(RunBatch) $(BestBatchBoyan); $(PlotBestBatchBoyan)
bestBatchRW:
	$(RunBatch) $(BestBatchRandomWalk); $(PlotBestBatchRW)

# --- PLOT RESULTS ---
plotBestBatch:
	$(PlotBestBatch)
plotBestBatchBaird:
	$(PlotBestBatchBaird)
plotBestBatchBoyan:
	$(PlotBestBatchBoyan)
plotBestBatchRW:
	$(PlotBestBatchRW)

# -------------- VIEW RESULTS -------------------
# I am too lazy to type filepaths in all the time
# -----------------------------------------------
openBairdFigs:
	gnome-open ./figures/Baird/*
openBoyanFigs:
	gnome-open ./figures/Boyan/*
openRWFigs:
	gnome-open ./figures/RandomWalk/*

# ============================================
# --- PARAMETER SWEEPS FOR EACH EXPERIMENT ---
# ============================================
sweepBaird:
	$(Run) $(Baird)
sweepBoyan:
	$(Run) $(Boyan)
sweepRW:
	$(Run) $(RandomWalk)
sweepOnline:
	$(Run) $(SweepOnline)

OnlinePath = ./experiments/online
BatchPath = ./experiments/batch

# ===================
# --- EXPERIMENTS ---
# ===================
# --- [[ ONLINE ]] ---
Baird = $(wildcard $(OnlinePath)/Baird/*.json)
Boyan = $(wildcard $(OnlinePath)/Boyan/*.json)
RandomWalk = $(wildcard $(OnlinePath)/RandomWalk/*.json)
SweepOnline = $(Baird) $(Boyan) $(RandomWalk)

BestBaird = $(wildcard $(OnlinePath)/Baird/best/*.json)
BestBoyan = $(wildcard $(OnlinePath)/Boyan/best/*.json)
RandomWalkDependent = $(wildcard $(OnlinePath)/RandomWalk/dependent/*.json)
RandomWalkInverted = $(wildcard $(OnlinePath)/RandomWalk/inverted/*.json)
RandomWalkTabular = $(wildcard $(OnlinePath)/RandomWalk/tabular/*.json)
BestRandomWalk = $(RandomWalkTabular) $(RandomWalkDependent) $(RandomWalkInverted)
BestOnline = $(BestBaird) $(BestBoyan) $(BestRandomWalk)

# --- [[ BATCH ]] ---
BatchBaird = $(wildcard $(BatchPath)/Baird/*.json)
BatchBoyan = $(wildcard $(BatchPath)/Boyan/*.json)
BatchRandomWalk = $(wildcard $(BatchPath)/RandomWalk/*.json)
SweepBatch = $(BatchBaird) $(BatchBoyan) $(BatchRandomWalk)

BestBatchBaird = $(wildcard $(BatchPath)/Baird/best/*.json)
BestBatchBoyan = $(wildcard $(BatchPath)/Boyan/best/*.json)
BatchRandomWalkDependent = $(wildcard $(BatchPath)/RandomWalk/dependent/*.json)
BatchRandomWalkInverted = $(wildcard $(BatchPath)/RandomWalk/inverted/*.json)
BatchRandomWalkTabular = $(wildcard $(BatchPath)/RandomWalk/tabular/*.json)
BestBatchRandomWalk = $(BatchRandomWalkTabular) $(BatchRandomWalkDependent) $(BatchRandomWalkInverted)
BestBatch = $(BestBatchBaird) $(BestBatchBoyan) $(BestBatchRandomWalk)

# ================
# --- COMMANDS ---
# ================
Py = python3.6
Run = $(Py) ./scripts/local.py ./src/main.py $(NUM_RUNS) ./results
RunBatch = $(Py) ./scripts/local.py ./src/batch.py $(NUM_RUNS) ./results
Plot = $(Py) ./analysis/learning_curve.py

# ----- PLOTTING -----
# --- [[ ONLINE ]] ---
PlotBestBaird = $(Py) $(OnlinePath)/Baird/best/learning_curve.py
PlotBestBoyan = $(Py) $(OnlinePath)/Boyan/best/learning_curve.py
PlotBestTabularRW = $(Py) $(OnlinePath)/RandomWalk/tabular/learning_curve.py
PlotBestInvertedRW = $(Py) $(OnlinePath)/RandomWalk/inverted/learning_curve.py
PlotBestDependentRW = $(Py) $(OnlinePath)/RandomWalk/dependent/learning_curve.py
PlotBestRW = $(PlotBestTabularRW); $(PlotBestInvertedRW); $(PlotBestDependentRW)
PlotBestOnline = $(PlotBestBaird); $(PlotBestBoyan); $(PlotBestRW)
# --- [[ BATCH ]] ---
PlotBestBatchBaird = $(Py) $(BatchPath)/Baird/best/learning_curve.py
PlotBestBatchBoyan = $(Py) $(BatchPath)/Boyan/best/learning_curve.py
PlotBestBatchTabularRW = $(Py) $(BatchPath)/RandomWalk/tabular/learning_curve.py
PlotBestBatchInvertedRW = $(Py) $(BatchPath)/RandomWalk/inverted/learning_curve.py
PlotBestBatchDependentRW = $(Py) $(BatchPath)/RandomWalk/dependent/learning_curve.py
PlotBestBatchRW = $(PlotBestBatchTabularRW); $(PlotBestBatchInvertedRW); $(PlotBestBatchDependentRW)
PlotBestBatch = $(PlotBestBatchBaird); $(PlotBestBatchBoyan); $(PlotBestBatchRW)
