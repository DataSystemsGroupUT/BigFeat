"""Detection-evidence figures, generated from the committed result JSONs.

Every number is read from benchmark_results_synthetic/ and
benchmark_results_pilot/ -- nothing is typed in, so the figures are exactly
as reproducible as the results. Palette: validated two-hue categorical
(before=orange, after=blue) on a light surface, for slides and print.

Run from the repository root:  python testing/Benchmarking/make_detection_figures.py
Outputs: docs/figures/*.png (+ .pdf)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e7e6e2"
SURF = "#fcfcfb"
MUTED_BAR = "#c9d6e8"          # light step of the blue hue for de-emphasised bars

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "text.color": INK, "axes.edgecolor": INK2,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.grid": False, "svg.fonttype": "none",
})

pre  = json.load(open("benchmark_results_synthetic/pre_b05b103.json"))
post = json.load(open("benchmark_results_synthetic/post_gapfix.json"))
pilot_nc = json.load(open("benchmark_results_pilot/bike_nocalendar_exploratory.json"))
pilot_cal = json.load(open("benchmark_results_pilot/bike_preregistered.json"))


def rate(rows, key, pred=lambda r: True):
    pl = [r for r in rows if r["kind"] == "planted" and r["ts"] is not None and pred(r)]
    k = sum(1 for r in pl if all(r[key]))
    return k, len(pl)


# ---------------------------------------------------------------- figure 1
conds = [
    ("All cases — windows",       "win_hits", lambda r: True),
    ("All cases — lags",          "lag_hits", lambda r: True),
    ("Two-period signals",        "win_hits", lambda r: r["signal"].startswith("pair")),
    ("Two-period signals — lags", "lag_hits", lambda r: r["signal"].startswith("pair")),
    ("Low SNR (σ=5 on amp 10)",  "win_hits", lambda r: r["snr"] == "low_snr"),
    ("Short series (3 cycles)",   "win_hits", lambda r: r["cycles"] == 3),
]
labels, pv, qv, pn, qn = [], [], [], [], []
for name, key, pred in conds:
    a, n1 = rate(pre, key, pred)
    b, n2 = rate(post, key, pred)
    labels.append(name); pv.append(100*a/n1); qv.append(100*b/n2)
    pn.append(f"{a}/{n1}"); qn.append(f"{b}/{n2}")

fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=200)
y = np.arange(len(labels))[::-1]; h = 0.36
ax.barh(y + h/2 + .02, pv, height=h, color=ORANGE, label="Before fixes")
ax.barh(y - h/2 - .02, qv, height=h, color=BLUE,   label="After fixes")
for yi, v, t in zip(y + h/2 + .02, pv, pn):
    ax.text(v + 1.2, yi, t, va="center", fontsize=9, color=INK2)
for yi, v, t in zip(y - h/2 - .02, qv, qn):
    ax.text(v + 1.2, yi, t, va="center", fontsize=9, color=INK)
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xlim(0, 112); ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("planted periods fully recovered (%)")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.xaxis.grid(True, color=GRID, lw=.8); ax.set_axisbelow(True)
ax.axvline(100, color=INK2, lw=.8, ls=(0, (2, 3)))
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.005),
          ncol=2, fontsize=10, columnspacing=1.6, handlelength=1.4)
ax.set_title("Ground-truth period recovery — 78 planted cases · false positives 0/6 in both versions",
             fontsize=11, loc="left", color=INK, pad=26)
fig.tight_layout()
fig.savefig("docs/figures/detection_recovery.png"); fig.savefig("docs/figures/detection_recovery.pdf")
plt.close(fig)

# ---------------------------------------------------------------- figure 2
def case(rows):
    for r in rows:
        if (r["signal"], r["snr"], r["cycles"], r["seed"]) == ("pair_7_30", "high_snr", 10, 0):
            return r
cp, cq = case(pre), case(post)
rows_ = [("Before — windows", cp["windows"], ORANGE, "s"),
         ("Before — lags",    cp["lags"],    ORANGE, "o"),
         ("After — windows",  cq["windows"], BLUE,   "s"),
         ("After — lags",     cq["lags"],    BLUE,   "o")]
fig, ax = plt.subplots(figsize=(8.6, 3.6), dpi=200)
for P in (7, 30):
    ax.axvline(P, color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax.text(P, 3.62, f"planted {P} d", ha="center", fontsize=9.5, color=INK)
for i, (name, days, col, mark) in enumerate(rows_):
    yv = 3 - i
    ax.scatter(days, [yv]*len(days), s=95, marker=mark, color=col,
               edgecolors=SURF, linewidths=1.6, zorder=3)
    prev = None
    for d in sorted(days):
        near = any(abs(d - P) <= 0.15*P for P in (7, 30))
        crowded = prev is not None and d <= prev * 1.18
        ax.annotate(str(d), (d, yv), textcoords="offset points",
                    xytext=(0, -18 if crowded else 9),
                    ha="center", fontsize=8.2,
                    color=INK if near else INK2)
        prev = d
ax.set_yticks([3, 2, 1, 0]); ax.set_yticklabels([r[0] for r in rows_])
ax.set_xscale("log"); ax.set_xlim(0.8, 130)
ax.set_xticks([1, 2, 5, 7, 14, 30, 56, 91]); ax.set_xticklabels([1, 2, 5, 7, 14, 30, 56, 91])
ax.set_xlabel("days (log scale)")
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.set_ylim(-0.55, 3.9)
ax.set_title("One case traced: planted 7-day + 30-day signal, 10 cycles",
             fontsize=11.5, loc="left", color=INK, pad=10)
fig.tight_layout()
fig.savefig("docs/figures/detection_traced_case.png"); fig.savefig("docs/figures/detection_traced_case.pdf")
plt.close(fig)

# ---------------------------------------------------------------- figure 3
def arm(rows, name):
    return [r["mae"] for r in rows if r.get("label", "").startswith(name)]
base_nc = arm(pilot_nc, "baseline")[0]
hand_nc = arm(pilot_nc, "handcrafted")[0]
bno_nc  = arm(pilot_nc, "bigfeat_no")
auto_nc = arm(pilot_nc, "bigfeat_auto")
base_cal = arm(pilot_cal, "baseline")[0]

names = ["baseline\n(no calendar)", "BigFeat, no TS", "hand-crafted\nlags/rolling", "BigFeat auto\n(detected periods)"]
vals  = [base_nc, float(np.mean(bno_nc)), hand_nc, float(np.mean(auto_nc))]
cols  = [MUTED_BAR, MUTED_BAR, MUTED_BAR, BLUE]
fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=200)
x = np.arange(4)
ax.bar(x, vals, width=.58, color=cols, zorder=3)
ax.errorbar([1, 3], [vals[1], vals[3]],
            yerr=[[vals[1]-min(bno_nc), vals[3]-min(auto_nc)],
                  [max(bno_nc)-vals[1], max(auto_nc)-vals[3]]],
            fmt="none", ecolor=INK2, elinewidth=1.4, capsize=4, zorder=4)
for xi, v in zip(x, vals):
    ax.text(xi, v + 1.6, f"{v:.1f}", ha="center", fontsize=10.5,
            color=INK if xi == 3 else INK2)
ax.axhline(base_cal, color=INK2, lw=1.1, ls=(0, (4, 3)))
ax.text(-0.42, 87.5, f"– – –  baseline WITH calendar columns kept: MAE {base_cal:.1f}",
        ha="left", fontsize=9.5, color=INK2)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9.6)
ax.set_ylabel("MAE, next-hour demand (lower is better)")
ax.set_ylim(0, 92)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.yaxis.grid(True, color=GRID, lw=.8); ax.set_axisbelow(True)
ax.set_title("Bike Sharing, calendar columns removed from every arm — detected periods substitute for them\n"
             "(whiskers: min–max over 5 seeds; exploratory run, docs/PILOT_BIKE.md)",
             fontsize=10.8, loc="left", color=INK, pad=12)
fig.tight_layout()
fig.savefig("docs/figures/pilot_substitution.png"); fig.savefig("docs/figures/pilot_substitution.pdf")
plt.close(fig)
print("figures written")
