"""Plot LPIPS / PSNR / PAC-S++ comparison bars. Writes into plots/."""
import os

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "plots"


def create_plot(data, title, color, filename, methods, x, y_min=None):
    fig, ax = plt.subplots(figsize=(4, 5))
    bars = ax.bar(x, data, color=color, width=0.7)
    ax.set_title(title, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    if y_min is None:
        ax.set_ylim(0, max(data) * 1.2)
    else:
        y_max = max(data)
        y_range = y_max - y_min
        ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.2)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 1),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    methods = [
        "Visible WM\n(size=30%,α=0.1)",
        "Visible WM\n(size=50%,α=0.1)",
        "Visible WM\n(size=50%,α=0.9)",
        "Invisible WM\nVINE",
        "Perturbation\nGlaze",
        "Perturbation\nMist",
    ]
    lpips = [0.07, 0.08, 0.16, 0.69, 0.13, 0.37]
    psnr = [27.03, 26.63, 22.37, 9.11, 24, 18.7]
    pac_spp = [-4.31, -4.11, -3.69, -4.01, 0.73, 2.32]
    colors = ["#1D3E56", "#6B8B92", "#A8C3CC"]
    x = np.arange(len(methods))

    create_plot(lpips, "LPIPS", colors[0], "lpips_plot.png", methods, x)
    create_plot(psnr, "PSNR", colors[1], "psnr_plot.png", methods, x)
    create_plot(pac_spp, "PAC-S++ (%Change)", colors[2], "pac_spp_plot.png", methods, x, min(pac_spp))