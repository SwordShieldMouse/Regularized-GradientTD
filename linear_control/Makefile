NUM_RUNS = 30

# use 'make <cmd>' to run the specified cmd
mountaincar:
	$(Run) $(MC); $(PlotMountainCar)
plot:
	$(PlotMountainCar)
open:
	gnome-open ./figures/MountainCar/*.png

# ===================
# --- EXPERIMENTS ---
# ===================
MC = $(wildcard ./experiments/MC-TC/*.json)

# ================
# --- COMMANDS ---
# ================
Py = python3.6
Run = $(Py) ./scripts/local.py ./src/main.py $(NUM_RUNS) ./results

PlotMountainCar = $(Py) ./experiments/MC-TC/learning_curve.py

