NUM_RUNS = 200
BATCH_RUNS = 500

# =========================================
# --- BEST SETTINGS FOR EACH EXPERIMENT ---
# =========================================
Online = $(Run) $(wildcard ./experiments/online/*/*/*.json)
Batch = $(RunSweep) $(wildcard ./experiments/batch/*/*/*.json)

all:
	$(Online); $(Batch)
online:
	$(Online)
batch:
	$(Batch)


# ========================
# --- PARAMETER SWEEPS ---
# ========================
OnlineSweep = $(Run) $(wildcard ./experiments/online/*/*.json)
BatchSweep = $(RunBatch) $(wildcard ./experiments/batch/*/*.json)

sweep:
	$(OnlineSweep); $(BatchSweep)
onlineSweep:
	$(OnlineSweep)
batchSweep:
	$(BatchSweep)

# ================
# --- PLOTTING ---
# ================

# -----------------------------
#     BEST PARAMETERS PLOTS	  
# -----------------------------
OnlinePlots = $(wildcard ./experiments/online/*/*/learning_curve.py)
PlotOnline = $(foreach path, $(OnlinePlots), $(Py) $(path);)

BatchPlots = $(wildcard ./experiments/batch/*/*/learning_curve.py)
PlotBatch = $(foreach path, $(BatchPlots), $(Py) $(path);)

plots:
	$(PlotOnline) $(PlotBatch)
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

sweepPlots:
	$(PlotOnlineSweep) $(PlotBatchSweep)
onlinePlots:
	$(PlotOnlineSweep)
batchPlots:
	$(PlotBatchSweep)
