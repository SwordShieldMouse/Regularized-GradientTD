NUM_RUNS = 200
BATCH_RUNS = 500

.PHONY : data plots sweepData sweepPlots results

# ========================================================
# ----------------------- COMMANDS -----------------------
# ========================================================
# Use these to generate the data/plots
# e.g. run `make <cmd>` at the command line, from the
# prediction subdirectory
# --------------------------------------------------------

# `make list`: print out a list of make targets available
.PHONY: list
list:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

# alias for `make results` because i'm lazy
all: results

# `make results` :: run all the experiments (best param settings)
# and then generate all the plots
results: data plots
# `make data` :: make the data using the best parameter settings
data: online batch
# `make plots` :: make the plots for the experiments
# using the best parameter settings
plots: onlinePlots batchPlots

# `make sweepResults` :: run the parameter sweeps for all experiments
# and generate the plots
sweepResults: sweepData sweepPlots
# `make sweepData` :: run the parameter sweeps
sweepData: onlineSweep batchSweep
# `make sweepPlots` :: make the plots for the data
# generated from the complete parameter sweep
sweepPlots: onlineSweepPlots batchSweepPlots

# ===========================================
# ---------------- UTILITIES ----------------
# ===========================================
# all the stuff below here is
# utility code defining the above commands
#
# You can also call the .PHONY targets
# defined here if you only want to generate
# a subset of the results
# (e.g. `make online`, `make batch` etc.)
# -------------------------------------------

# Commands that we'll run to generate data
Py = python3.6
Run = $(Py) ./scripts/local.py ./src/main.py $(NUM_RUNS) ./results
RunBatch = $(Py) ./scripts/local.py ./src/batch.py $(BATCH_RUNS) ./results

# -----------------------------------------
# --- BEST SETTINGS FOR EACH EXPERIMENT ---
# -----------------------------------------
Online = $(wildcard ./experiments/online/*/*/*.json)
Batch = $(wildcard ./experiments/batch/*/*/*.json)

.PHONY : online batch
online:
	$(Run) $(Online)
batch:
	$(RunBatch) $(Batch)


# ------------------------
# --- PARAMETER SWEEPS ---
# ------------------------
OnlineSweep = $(wildcard ./experiments/online/*/*.json)
BatchSweep = $(wildcard ./experiments/batch/*/*.json)

.PHONY : onlineSweep batchSweep
onlineSweep:
	$(Run) $(OnlineSweep)
batchSweep:
	$(RunBatch) $(BatchSweep)

# ----------------
# --- PLOTTING ---
# ----------------

# -----------------------------
#     BEST PARAMETERS PLOTS	  
# -----------------------------
OnlinePlots = $(wildcard ./experiments/online/*/*/learning_curve.py)
PlotOnline = $(foreach path, $(OnlinePlots), $(Py) $(path);)

BatchPlots = $(wildcard ./experiments/batch/*/*/learning_curve.py)
PlotBatch = $(foreach path, $(BatchPlots), $(Py) $(path);)

.PHONY : onlinePlots batchPlots
onlinePlots:
	$(PlotOnline)
batchPlots:
	$(PlotBatch)

# -----------------------------
#     PARAMETER SWEEP PLOTS
# -----------------------------
OnlineSweepPlots = $(wildcard ./experiments/online/*/learning_curve.py)
PlotOnlineSweep = $(foreach path, $(OnlineSweepPlots), $(Py) $(path);)

BatchSweepPlots = $(wildcard ./experiments/batch/*/learning_curve.py)
PlotBatchSweep = $(foreach path, $(BatchSweepPlots), $(Py) $(path);)

.PHONY : onlineSweepPlots batchSweepPlots
onlineSweepPlots:
	$(PlotOnlineSweep)
batchSweepPlots:
	$(PlotBatchSweep)
