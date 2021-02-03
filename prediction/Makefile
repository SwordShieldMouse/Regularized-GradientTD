# ==================
# --- QUICKSTART ---
# ==================
#
# Main Commands
# -------------
# - `make list`: print the available make targets
#
# - `make` or `make all`: run all the experiments and make all the plots
# - `make data`: run all the experiments
# - `make plots`: make all the plots
#
# Sub-commands
# ------------
# - `make online`: run MDP experiments with online evaluation
# - `make batch`: run MDP the experiments with batch evaluation
# - `make nexting`: run the multi-timescale nexting experiment
#
# - `make mdpPlots`: plot the mdp experiments (both batch and online)
# - `make nextingPlots`: plot the nexting experiment

.PHONY : data plots results

# ========================================================
# ----------------------- COMMANDS -----------------------
# ========================================================
# Use these to generate the data/plots
# e.g. run `make <cmd>` at the command line, from the
# prediction subdirectory
# --------------------------------------------------------

# `make` or `make all` :: run all the experiments (best param settings)
# and then generate all the plots
all: data plots

# `make data` :: make the data using the best parameter settings
data: online batch nexting

# `make plots` :: make the plots for the experiments
# using the best parameter settings
plots: barPlot mdpPlots cdfPlots waterfallPlots nextingPlots

# `make list`: print out a list of make targets available
# lol shut up it works fine
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

NUM_RUNS = 200

# Commands that we'll run to generate data
Py = python3.6
Run = $(Py) ./scripts/local.py ./src/main.py $(NUM_RUNS) ./results
RunBatch = $(Py) ./scripts/local.py ./src/batch.py $(NUM_RUNS) ./results
RunNexting = $(Py) ./scripts/local.py ./src/nexting.py 1 ./results

# -------------------
# --- EXPERIMENTS ---
# -------------------
.PHONY : online batch nexting
online:
	$(Run) $(foreach exp, Baird Boyan RandomWalk, $(wildcard ./experiments/online/$(path)/*.json))
batch:
	$(RunBatch) $(foreach exp, Baird Boyan RandomWalk, $(wildcard ./experiments/batch/$(path)/*.json)
nexting:
	$(RunNexting) $(wildcard ./experiments/online/Nexting/*/*.json) $(wildcard ./experiments/*/*/*.json)

# ----------------
# --- PLOTTING ---
# ----------------

# -----------------------------
#     BEST PARAMETERS PLOTS	  
# -----------------------------

PF_ALGS =  cwpfgtd pfgtd pfgtd+ 
BAIRD_BASELINES = tdc tdrc gtd2
BASELINES = td $(BAIRD_BASELINES)

BAIRD_ALGS = $(BAIRD_BASELINES) $(PF_ALGS)
ALL_ALGS = $(BASELINES) $(PF_ALGS)

#SETTINGS = online batch
SETTINGS = online

.PHONY : mdpPlots nextingPlots waterfallPlots waterfallPlots cdfPlots
barPlot:
	$(Py) analysis/barplots.py
mdpPlots:
	$(foreach setting, $(SETTINGS), $(foreach exp, Boyan RandomWalk, $(Py) ./analysis/mdp_plots.py $(foreach alg, $(ALL_ALGS), ./experiments/$(setting)/$(exp)/$(alg).json);))
	$(foreach setting, $(SETTINGS), $(Py) ./analysis/mdp_plots.py $(foreach alg, $(BAIRD_ALGS), ./experiments/$(setting)/Baird/$(alg).json);))
cdfPlots:
	$(foreach exp, Boyan RandomWalk, $(Py) ./analysis/cdf_plots.py $(foreach alg, $(BASELINES), ./experiments/cdf/$(exp)/$(alg).json) $(foreach alg, $(PF_ALGS), ./experiments/cdf/$(exp)/$(alg).json);)
	$(Py) ./analysis/cdf_plots.py $(foreach alg, $(BAIRD_BASELINES), ./experiments/cdf/Baird/$(alg).json) $(foreach alg, $(PF_ALGS), ./experiments/cdf/Baird/$(alg).json)
	$(foreach exp, Boyan RandomWalk, $(Py) ./analysis/cdf_plots.py $(foreach alg, $(BASELINES), ./experiments/cdf_grid/$(exp)/$(alg).json) $(foreach alg, $(PF_ALGS), ./experiments/cdf/$(exp)/$(alg).json);)
	$(Py) ./analysis/cdf_plots.py $(foreach alg, $(BAIRD_BASELINES), ./experiments/cdf_grid/Baird/$(alg).json) $(foreach alg, $(PF_ALGS), ./experiments/cdf/Baird/$(alg).json)
waterfallPlots:
	$(foreach exp, Boyan RandomWalk, $(Py) ./analysis/waterfall.py $(foreach alg, $(ALL_ALGS), ./experiments/cdf/$(exp)/$(alg).json);)
	$(Py) ./analysis/waterfall.py $(foreach alg, $(BAIRD_ALGS), ./experiments/cdf/Baird/$(alg).json)
nextingPlots:
	$(Py) ./analysis/nexting_plots.py $(foreach exp, $(ALL_ALGS), ./experiments/online/Nexting/lambda0.0/$(exp).json)
	# mv figures/Nexting/lambda0.0 ./figures/Nexting/main
	# $(Py) ./analysis/nexting_plots.py $(foreach exp, gtd2_etaSweep_full gtd2_etaSweep_half gtd2, ./experiments/online/Nexting/lambda0.0/$(exp).json)
	# mv figures/Nexting/lambda0.0 ./figures/Nexting/appendix
