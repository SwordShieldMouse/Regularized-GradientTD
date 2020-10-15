NUM_RUNS = 200
NULL_IDX = 0

OnlinePath = ./experiments/online

# ===================
# --- EXPERIMENTS ---
# ===================
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

# ================
# --- COMMANDS ---
# ================
Py = python3.6
Run = $(Py) ./scripts/local.py ./src/main.py $(NUM_RUNS) ./results
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
# TODO

# =========================================
# --- BEST SETTINGS FOR EACH EXPERIMENT ---
# =========================================

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
