# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Render the evidence figures for Megatron-LM issue #7857.

The figures use the exact numbers the issue reports (Nemotron-H-8B-Base-8K,
generated-token index 29, seed 123, temperature=0.1, top_p=0.9) plus the
FP16 zero-temperature probe.

Run inside the Megatron environment:

    python make_figures.py --out-dir .

* ``fig1_precision.png``  -- the two failures the issue reports.
* ``fig2_disagreement.png`` -- how often the pre-fix path disagrees with the
  FP32 reference over the (temperature, logit-gap) plane.
"""

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from megatron.core.inference.sampling.torch_sampling import TorchSampling

BLUE = "#2f6fb0"
ORANGE = "#d1732a"
GREY = "#8a8f98"
DARK = "#333333"

# Issue #7857, step 29 of the Nemotron-H-8B email prompt.
LEADING_LOGIT = 26.625
RUNNER_UP_LOGIT = 26.375
LEADING_TOKEN = 1317  # " to"
RUNNER_UP_TOKEN = 2136  # " over"
TOP_P = 0.9
TEMPERATURE = 0.1

MIN_SAMPLING_TEMPERATURE = 1e-6
FP16_MAX = 65504.0


def prefix_filter_logits(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """The pre-fix filter: divide in the input dtype, then mask the nucleus.

    Mirrors the pre-fix ``TorchSampling.filter_logits`` (clone, in-dtype
    ``div_``, then the same nucleus mask) so the figures show what the
    released code did.
    """
    filtered = logits.clone()
    if temperature != 1.0:
        filtered.div_(max(temperature, MIN_SAMPLING_TEMPERATURE))
    sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
    cumulative = sorted_logits.softmax(dim=-1, dtype=torch.float32).cumsum(dim=-1)
    mask = cumulative > top_p
    mask[:, 1:] = mask[:, :-1].clone()
    mask[..., 0] = 0
    mask = mask.scatter(1, sorted_indices, mask)
    return filtered.masked_fill(mask, float("-inf"))


def analyse(logits: torch.Tensor, temperature: float, top_p: float, *, prefix: bool):
    """Return (pre-filter probabilities, nucleus size).

    ``prefix=True`` reproduces the pre-fix path (in-dtype divisor), otherwise
    the fixed ``TorchSampling.filter_logits`` is used.
    """
    effective = max(temperature, MIN_SAMPLING_TEMPERATURE)
    if prefix:
        scaled = logits.clone()
        scaled.div_(effective)
        filtered = prefix_filter_logits(logits, temperature, top_p)
    else:
        # The fixed path scales in fp32 and then masks the nucleus.
        scaled = logits.to(torch.float32).clone()
        scaled.div_(effective)
        filtered = TorchSampling.filter_logits(logits, temperature, top_k=0, top_p=top_p)
    probs = scaled.softmax(dim=-1, dtype=torch.float32)
    return probs[0].tolist(), int(torch.isfinite(filtered).sum())


def figure_one(out_path: str) -> None:
    bf16 = torch.tensor(
        [[LEADING_LOGIT, RUNNER_UP_LOGIT] + [-100.0] * 126], dtype=torch.bfloat16
    )

    prefix_probs, prefix_nucleus = analyse(bf16, TEMPERATURE, TOP_P, prefix=True)
    fixed_probs, fixed_nucleus = analyse(bf16, TEMPERATURE, TOP_P, prefix=False)

    fig, (ax_probs, ax_overflow) = plt.subplots(1, 2, figsize=(13.0, 5.6))

    # ---------------------------------------------------------------- panel A
    labels = [f'token {LEADING_TOKEN}\n" to"', f'token {RUNNER_UP_TOKEN}\n" over"']
    positions = np.arange(2)
    width = 0.36
    ax_probs.bar(
        positions - width / 2,
        prefix_probs[:2],
        width,
        label="pre-fix: BF16 divisor",
        color=ORANGE,
        edgecolor="white",
    )
    ax_probs.bar(
        positions + width / 2,
        fixed_probs[:2],
        width,
        label="fixed: FP32 divisor",
        color=BLUE,
        edgecolor="white",
    )
    ax_probs.axhline(TOP_P, color=GREY, linestyle="--", linewidth=1.2)
    ax_probs.text(
        1.58,
        TOP_P + 0.02,
        f"top_p = {TOP_P}",
        color=GREY,
        fontsize=9,
        ha="right",
    )
    for x, value in zip(positions - width / 2, prefix_probs[:2]):
        ax_probs.text(x, value + 0.018, f"{value:.6f}", ha="center", fontsize=9, color=ORANGE)
    for x, value in zip(positions + width / 2, fixed_probs[:2]):
        ax_probs.text(x, value + 0.018, f"{value:.6f}", ha="center", fontsize=9, color=BLUE)
    ax_probs.set_xticks(positions)
    ax_probs.set_xticklabels(labels)
    ax_probs.set_xlim(-0.6, 1.6)
    ax_probs.set_ylim(0, 1.0)
    ax_probs.set_ylabel("probability before nucleus filtering")
    ax_probs.set_title(
        "A. BF16 rounding of the temperature divisor moves the top-p boundary\n"
        f"Nemotron-H-8B, step 29, temperature={TEMPERATURE}, top_p={TOP_P}",
        fontsize=10.5,
    )
    ax_probs.legend(
        frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=9.5
    )
    ax_probs.spines[["top", "right"]].set_visible(False)

    # ---------------------------------------------------------------- panel B
    logit = 30.0
    temperatures = np.logspace(-6, 0.3, 900)
    exact = logit / temperatures
    crossing = logit / FP16_MAX

    ax_overflow.plot(
        temperatures,
        exact,
        color=BLUE,
        linewidth=2.0,
        label="FP32 divisor (fixed): stays finite",
    )
    ax_overflow.plot(
        temperatures[: int(np.searchsorted(temperatures, crossing)) + 1],
        exact[: int(np.searchsorted(temperatures, crossing)) + 1],
        color=ORANGE,
        linewidth=2.4,
        label="FP16 divisor (pre-fix): finite until it overflows",
    )
    ax_overflow.fill_between(
        temperatures,
        FP16_MAX,
        1e9,
        where=temperatures <= crossing,
        color=ORANGE,
        alpha=0.13,
    )
    ax_overflow.axhline(FP16_MAX, color=ORANGE, linestyle="--", linewidth=1.2)
    ax_overflow.text(
        1.6,
        FP16_MAX * 2.2,
        "FP16 max = 65504",
        color=ORANGE,
        fontsize=9,
        ha="right",
    )
    ax_overflow.axvline(MIN_SAMPLING_TEMPERATURE, color=GREY, linestyle=":", linewidth=1.2)
    ax_overflow.text(
        MIN_SAMPLING_TEMPERATURE * 1.4,
        1e7,
        "temperature floor 1e-6",
        color=GREY,
        fontsize=9,
    )
    ax_overflow.axvline(crossing, color=ORANGE, linestyle="-.", linewidth=1.0)
    ax_overflow.annotate(
        f"FP16 overflows below T = {crossing:.2e}\n-> inf, then a NaN softmax",
        xy=(crossing, FP16_MAX),
        xytext=(3e-5, 4e7),
        fontsize=9,
        color=ORANGE,
        arrowprops={"arrowstyle": "->", "color": ORANGE, "linewidth": 1.0},
    )
    ax_overflow.text(
        2.5e-6,
        1.2e8,
        "inf",
        color=ORANGE,
        fontsize=11,
        fontweight="bold",
    )
    ax_overflow.set_xscale("log")
    ax_overflow.set_yscale("log")
    ax_overflow.set_xlim(1e-6, 2e0)
    ax_overflow.set_ylim(1e-2, 3e8)
    ax_overflow.set_xlabel("temperature")
    ax_overflow.set_ylabel("scaled logit magnitude  (logit / temperature)")
    ax_overflow.set_title(
        "B. FP16 scaling overflows to inf at low temperature\n"
        "logit = 30.0, temperature floored at 1e-6",
        fontsize=10.5,
    )
    ax_overflow.legend(
        frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, fontsize=9.5
    )
    ax_overflow.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.text(
        0.5,
        0.012,
        f"pre-fix nucleus = {{{LEADING_TOKEN}, {RUNNER_UP_TOKEN}}} ({prefix_nucleus} tokens)"
        f" -> the engine can emit {RUNNER_UP_TOKEN}   |   "
        f"fixed nucleus = {{{LEADING_TOKEN}}} ({fixed_nucleus} token)"
        f" -> the engine emits {LEADING_TOKEN}",
        ha="center",
        fontsize=10,
        color=DARK,
    )
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def figure_two(out_path: str) -> None:
    fig, (ax_curve, ax_map) = plt.subplots(1, 2, figsize=(13.0, 5.2))

    # ------------------------------------------------------- panel A: curves
    gaps = np.linspace(0.0, 0.8, 801)
    # FP32 reference: the gap is exact (no low-precision rounding).
    fp32_logits = torch.tensor(
        [[LEADING_LOGIT, LEADING_LOGIT - float(gap)] for gap in gaps], dtype=torch.float32
    )
    fp32_probs = fp32_logits.softmax(dim=-1)[:, 1].numpy()
    # Pre-fix path: the gap itself is rounded to bf16 before the division, and
    # the scaled logits are rounded again by the bf16 divisor.
    bf16_logits = torch.tensor(
        [[LEADING_LOGIT, LEADING_LOGIT - float(gap)] for gap in gaps], dtype=torch.bfloat16
    )
    bf16_probs = (bf16_logits / TEMPERATURE).float().softmax(dim=-1)[:, 1].numpy()

    fp32_cross = TEMPERATURE * math.log(1.0 / (1.0 - TOP_P))
    crossing = gaps[np.argmax(bf16_probs <= 1.0 - TOP_P)]

    ax_curve.axvspan(fp32_cross, crossing, color=ORANGE, alpha=0.10)
    ax_curve.plot(gaps, fp32_probs, color=BLUE, linewidth=2.0, label="FP32 divisor (reference)")
    ax_curve.plot(
        gaps,
        bf16_probs,
        color=ORANGE,
        linewidth=1.8,
        drawstyle="steps-post",
        label="BF16 divisor (pre-fix)",
    )
    ax_curve.axhline(1.0 - TOP_P, color=GREY, linestyle="--", linewidth=1.2)
    ax_curve.text(
        0.79,
        1.0 - TOP_P + 0.006,
        f"runner-up mass limit = {1.0 - TOP_P:.1f}",
        color=GREY,
        fontsize=9,
        ha="right",
    )
    ax_curve.axvline(fp32_cross, color=BLUE, linestyle="-.", linewidth=1.0)
    ax_curve.axvline(crossing, color=ORANGE, linestyle="-.", linewidth=1.0)
    ax_curve.annotate(
        f"FP32 drops the runner-up\nabove a gap of {fp32_cross:.2f}",
        xy=(fp32_cross, 0.30),
        xytext=(0.40, 0.40),
        fontsize=9,
        color=BLUE,
        arrowprops={"arrowstyle": "->", "color": BLUE, "linewidth": 1.0},
    )
    ax_curve.annotate(
        f"BF16 keeps it until a gap of {crossing:.2f}\n(shaded band = divergent)",
        xy=(crossing, 0.22),
        xytext=(0.44, 0.22),
        fontsize=9,
        color=ORANGE,
        arrowprops={"arrowstyle": "->", "color": ORANGE, "linewidth": 1.0},
    )
    ax_curve.set_xlabel("logit gap between the leading and the runner-up token")
    ax_curve.set_ylabel("runner-up probability")
    ax_curve.set_ylim(0, 0.52)
    ax_curve.set_title(
        "A. The BF16 divisor shifts the nucleus boundary\n"
        f"leading logit {LEADING_LOGIT}, temperature {TEMPERATURE}",
        fontsize=10.5,
    )
    ax_curve.legend(
        frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=9.5
    )
    ax_curve.spines[["top", "right"]].set_visible(False)

    # ------------------------------------------------------- panel B: heatmap
    temperatures = np.logspace(-2.0, 0.3, 100)
    gap_grid = np.linspace(0.0, 0.6, 100)
    disagreement = np.zeros((temperatures.shape[0], gap_grid.shape[0]))

    for i, temperature in enumerate(temperatures):
        for j, gap in enumerate(gap_grid):
            logits = torch.tensor(
                [[LEADING_LOGIT, LEADING_LOGIT - gap] + [-100.0] * 126], dtype=torch.bfloat16
            )
            prefix_probs, prefix_nucleus = analyse(logits, float(temperature), TOP_P, prefix=True)
            _, fixed_nucleus = analyse(logits, float(temperature), TOP_P, prefix=False)
            if prefix_nucleus > fixed_nucleus and prefix_probs[1] > 0.0:
                # Probability the pre-fix path draws a token the FP32
                # reference excludes (Gumbel-max == multinomial).
                disagreement[i, j] = prefix_probs[1] / (prefix_probs[0] + prefix_probs[1])

    mesh = ax_map.pcolormesh(
        gap_grid,
        temperatures,
        disagreement,
        cmap="YlOrRd",
        shading="auto",
        vmin=0.0,
        vmax=0.5,
    )
    issue_gap = LEADING_LOGIT - RUNNER_UP_LOGIT
    ax_map.plot(
        [issue_gap],
        [TEMPERATURE],
        marker="o",
        markersize=9,
        markerfacecolor="none",
        markeredgecolor="#1b3a5c",
        markeredgewidth=2.0,
    )
    ax_map.annotate(
        "issue #7857\n(temperature=0.1, gap=0.25)",
        xy=(issue_gap, TEMPERATURE),
        xytext=(issue_gap + 0.09, TEMPERATURE * 0.35),
        fontsize=9,
        color="#1b3a5c",
        arrowprops={"arrowstyle": "->", "color": "#1b3a5c", "linewidth": 1.0},
    )
    ax_map.set_yscale("log")
    ax_map.set_xlabel("logit gap between the leading and the runner-up token")
    ax_map.set_ylabel("temperature")
    ax_map.set_title(
        "B. Probability the pre-fix path emits a token the FP32 nucleus excludes\n"
        "BF16 logits, top_p = 0.9 (dark = the engine can diverge)",
        fontsize=10.5,
    )
    colorbar = fig.colorbar(mesh, ax=ax_map)
    colorbar.set_label("probability of divergence", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    figure_one(os.path.join(args.out_dir, "fig1_precision.png"))
    figure_two(os.path.join(args.out_dir, "fig2_disagreement.png"))
    print("wrote fig1_precision.png and fig2_disagreement.png")


if __name__ == "__main__":
    main()
