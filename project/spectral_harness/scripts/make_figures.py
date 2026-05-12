"""Build the three loss-vs-step figures and the wall-clock table for the
Spectral Optimizer Capstone report.

Reads:
    logs/grid_task_a_full.csv
    logs/grid_task_b_full.csv
    logs/grid_task_c_full.csv

For each task, picks one (lr, wd, eps) configuration -- the best-of-grid
config by mean final loss across the three seeds, common to AdamW and the
spectral optimizer where possible -- and plots median + (min, max) over
seeds for both optimizers as a function of step.

Writes:
    figures/fig_task_a.pdf
    figures/fig_task_b.pdf
    figures/fig_task_c.pdf
    figures/wallclock.csv

Also prints a wall-clock summary table to stdout.

Usage:
    python scripts/make_figures.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load(task: str) -> pd.DataFrame:
    df = pd.read_csv(LOG_DIR / f"grid_{task}_full.csv")
    # Drop any non-finite rows from early-aborted runs.
    df = df[np.isfinite(df["loss"])].copy()
    return df


# --------------------------------------------------------------------- #
# Pick "best-of-grid" config for each task.  Definition: the
# (lr, wd, eps) tuple whose mean across seeds of the FINAL logged loss
# is the lowest, computed once per optimizer.  For the figure we show
# the AdamW best-config curve and the spectral best-config curve; if
# they pick the same config (often the case), the comparison is matched.
# --------------------------------------------------------------------- #
def best_config(df: pd.DataFrame, optimizer: str) -> tuple[float, float, float]:
    """Return (lr, wd, eps) with the lowest mean-over-seeds final loss."""
    sub = df[df["optimizer"] == optimizer]
    finals = (sub.sort_values("step")
              .groupby(["lr", "wd", "eps", "seed"], as_index=False)
              .tail(1))
    summary = (finals.groupby(["lr", "wd", "eps"], as_index=False)["loss"]
               .mean()
               .sort_values("loss"))
    row = summary.iloc[0]
    return float(row["lr"]), float(row["wd"]), float(row["eps"])


def curve(df: pd.DataFrame, optimizer: str, lr: float, wd: float, eps: float
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (steps, median_loss, min_loss, max_loss) across seeds."""
    sub = df[(df["optimizer"] == optimizer)
             & np.isclose(df["lr"], lr)
             & np.isclose(df["wd"], wd)
             & np.isclose(df["eps"], eps)]
    pivot = sub.pivot_table(index="step", columns="seed", values="loss",
                            aggfunc="first").sort_index()
    steps = pivot.index.to_numpy()
    vals = pivot.to_numpy()
    return steps, np.median(vals, axis=1), vals.min(axis=1), vals.max(axis=1)


def wallclock(df: pd.DataFrame, optimizer: str, lr: float, wd: float, eps: float
              ) -> float:
    """Return median (over seeds) of the cumulative ms at the final
    logged step."""
    sub = df[(df["optimizer"] == optimizer)
             & np.isclose(df["lr"], lr)
             & np.isclose(df["wd"], wd)
             & np.isclose(df["eps"], eps)]
    finals = (sub.sort_values("step")
              .groupby("seed", as_index=False).tail(1))
    return float(np.median(finals["ms"].to_numpy()))


# --------------------------------------------------------------------- #
# Plot one task panel.
# --------------------------------------------------------------------- #
def plot_task(task: str, label: str, y_floor: float, out_path: Path) -> dict:
    df = load(task)

    adamw_cfg = best_config(df, "adamw")
    spec_cfg = best_config(df, "spectral")

    print(f"[{task}] best AdamW cfg (lr,wd,eps) = {adamw_cfg}")
    print(f"[{task}] best spec  cfg (lr,wd,eps) = {spec_cfg}")

    fig, ax = plt.subplots(figsize=(5.5, 3.4))

    s_a, m_a, lo_a, hi_a = curve(df, "adamw", *adamw_cfg)
    s_s, m_s, lo_s, hi_s = curve(df, "spectral", *spec_cfg)

    ax.fill_between(s_a, np.maximum(lo_a, y_floor),
                    np.maximum(hi_a, y_floor),
                    alpha=0.18, color="#1f77b4")
    ax.plot(s_a, np.maximum(m_a, y_floor), color="#1f77b4",
            lw=2.0,
            label=f"AdamW (lr={adamw_cfg[0]:g}, wd={adamw_cfg[1]:g}, eps={adamw_cfg[2]:g})")

    ax.fill_between(s_s, np.maximum(lo_s, y_floor),
                    np.maximum(hi_s, y_floor),
                    alpha=0.18, color="#d62728")
    ax.plot(s_s, np.maximum(m_s, y_floor), color="#d62728",
            lw=2.0, ls="--",
            label=f"Spectral (lr={spec_cfg[0]:g}, wd={spec_cfg[1]:g}, eps={spec_cfg[2]:g})")

    ax.set_yscale("log")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss (median; band = [min, max] over 3 seeds)")
    ax.set_title(label)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return {
        "task": task,
        "adamw_cfg": adamw_cfg,
        "spec_cfg": spec_cfg,
        "adamw_final_median": float(m_a[-1]),
        "spec_final_median": float(m_s[-1]),
        "adamw_ms_median": wallclock(df, "adamw", *adamw_cfg),
        "spec_ms_median": wallclock(df, "spectral", *spec_cfg),
    }


def main() -> None:
    rows = [
        plot_task("task_a", "Task A: FashionMNIST residual MLP",
                  y_floor=0.05, out_path=FIG_DIR / "fig_task_a.pdf"),
        plot_task("task_b", r"Task B: rotated anisotropy, $\kappa=10^3$",
                  y_floor=1e-4, out_path=FIG_DIR / "fig_task_b.pdf"),
        plot_task("task_c", r"Task C: teacher--student, $\alpha=0.3$",
                  y_floor=1e-4, out_path=FIG_DIR / "fig_task_c.pdf"),
    ]
    summary = pd.DataFrame(rows)
    summary["spec_overhead_ratio"] = (
        summary["spec_ms_median"] / summary["adamw_ms_median"])
    summary.to_csv(FIG_DIR / "wallclock.csv", index=False)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
