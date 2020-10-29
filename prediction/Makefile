NUM_RUNS = 200
BATCH_RUNS = 200

.PHONY : data plots results

# ========================================================
# ----------------------- COMMANDS -----------------------
# ========================================================
# Use these to generate the data/plots
# e.g. run `make <cmd>` at the command line, from the
# prediction subdirectory
# --------------------------------------------------------

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

# `make list`: print out a list of make targets available
# lol shut up it works fine
.PHONY: list
list:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'


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
RunBatch = $(Py) ./scripts/local.py ./src/batch.py $(NUM_RUNS) ./results

# -------------------
# --- EXPERIMENTS ---
# -------------------
.PHONY : online batch
online:
	$(Run) $(wildcard ./experiments/online/*/*.json)
batch:
	$(RunBatch) $(wildcard ./experiments/batch/*/*.json)

# ----------------
# --- PLOTTING ---
# ----------------

# -----------------------------
#     BEST PARAMETERS PLOTS	  
# -----------------------------
OnlinePlots = $(wildcard ./experiments/online/*/learning_curve.py)
BatchPlots = $(wildcard ./experiments/batch/*/learning_curve.py)

.PHONY : onlinePlots batchPlots
onlinePlots:
	$(foreach path, $(OnlinePlots), $(Py) $(path);)
batchPlots:
	$(foreach path, $(BatchPlots), $(Py) $(path);)
