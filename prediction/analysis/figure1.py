# ---------------------
# Plotting the bar plot
# ---------------------
import matplotlib.pyplot as plt

ax = plt.gca()
f = plt.gcf()

# get TDRC's baseline performance for each problem
baselines = [None] * len(PROBLEMS)
for i, problem in enumerate(PROBLEMS):
    env = problem['env'].__name__
    rep = problem['representation'].__name__

    mean_curve, _, _ = collector.getStats(f'{env}-{rep}-TDRC')

    # compute TDRC's AUC
    baselines[i] = mean_curve.mean()

# how far from the left side of the plot to put the bar
offset = -3
for i, problem in enumerate(PROBLEMS):
    # additional offset between problems
    # creates space between the problems
    offset += 3
    for j, Learner in enumerate(LEARNERS):
        learner = Learner.__name__
        env = problem['env'].__name__
        rep = problem['representation'].__name__

        x = i * len(LEARNERS) + j + offset

        mean_curve, stderr_curve, runs = collector.getStats(f'{env}-{rep}-{learner}')
        auc = mean_curve.mean()
        auc_stderr = stderr_curve.mean()

        relative_auc = auc / baselines[i]
        relative_stderr = auc_stderr / baselines[i]

        ax.bar(x, relative_auc, yerr=relative_stderr, color=COLORS[learner], tick_label='')

# plt.show()
fig_dir = "figures/prediction/"
os.makedirs(fig_dir, exist_ok=True)
plt.savefig(f"{fig_dir}auc.png")

# =========================
# --- FINAL PERFORMANCE ---
# =========================
f, ax = plt.subplots()

# get TDRC's baseline performance for each problem
baselines = [None] * len(PROBLEMS)
for i, problem in enumerate(PROBLEMS):
    env = problem['env'].__name__
    rep = problem['representation'].__name__

    mean_curve, _, _ = collector.getStats(f'{env}-{rep}-TDRC')

    # compute TDRC's AUC
    baselines[i] = mean_curve.mean()

# how far from the left side of the plot to put the bar
offset = -3
for i, problem in enumerate(PROBLEMS):
    # additional offset between problems
    # creates space between the problems
    offset += 3
    for j, Learner in enumerate(LEARNERS):
        learner = Learner.__name__
        env = problem['env'].__name__
        rep = problem['representation'].__name__

        x = i * len(LEARNERS) + j + offset

        mean_curve, stderr_curve, runs = collector.getStats(f'{env}-{rep}-{learner}')
        auc = mean_curve[-1]
        auc_stderr = stderr_curve[-1]

        relative_auc = auc / baselines[i]
        relative_stderr = auc_stderr / baselines[i]

        ax.bar(x, relative_auc, yerr=relative_stderr, color=COLORS[learner], tick_label='')

# plt.show()
fig_dir = "figures/prediction/"
os.makedirs(fig_dir, exist_ok=True)
plt.savefig(f"{fig_dir}final.png")

# =======================
# --- LEARNING CURVES ---
# =======================
for i, problem in enumerate(PROBLEMS):
    # additional offset between problems
    # creates space between the problems
    fig, ax = plt.subplots()

    env = problem['env'].__name__
    rep = problem['representation'].__name__
    for j, Learner in enumerate(LEARNERS):
        learner = Learner.__name__

        mean_curve, stderr_curve, runs = collector.getStats(f'{env}-{rep}-{learner}')

        ax.plot(np.arange(len(mean_curve)), mean_curve, color=COLORS[learner], label=learner)
        ax.fill_between(np.arange(len(mean_curve)), mean_curve - stderr_curve, mean_curve + stderr_curve, alpha=0.2, color=COLORS[learner])

    ax.set_title(f"{env} {rep}")
    ax.legend()

    fig_dir = "figures/prediction/learning_curves/"
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(f"{fig_dir}{env}_{rep}.png")

# ========================================
# --- LEARNING CURVES (INDIVIDUAL RUNS)---
# ========================================
for i, problem in enumerate(PROBLEMS):
    # additional offset between problems
    # creates space between the problems
    fig, ax = plt.subplots()

    env = problem['env'].__name__
    rep = problem['representation'].__name__
    for j, Learner in enumerate(LEARNERS):
        learner = Learner.__name__

        all_data = collector.all_data[f'{env}-{rep}-{learner}']

        for d in all_data:
            data = np.array(d)
            ax.plot(np.arange(len(d)), d,  color=COLORS[learner], alpha=0.2)

        mean_curve = np.mean(np.array(all_data), axis=0)
        ax.plot(np.arange(len(mean_curve)), mean_curve,  color=COLORS[learner], label=learner, linewidth=2.0)

    ax.legend()
    ax.set_title(f"{env} {rep}")

    fig_dir = "figures/prediction/learning_curves/"
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(f"{fig_dir}{env}_{rep}_allRuns.png")
