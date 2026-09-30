"""Annotated rectangular parameter matrices for reproducible study figures."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def save_matrix_panels(
    path: Path, rows: list[dict[str, Any]], levels: list[float], fields: list[tuple[str, str, str]], *, title: str
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for ax, (key, label, fmt) in zip(axes.flat, fields):
        matrix = np.full((len(levels), len(levels)), np.nan)
        for row in rows:
            if row.get(key) is not None:
                i, j = levels.index(row["front_rc_mm"]), levels.index(row["rear_rc_mm"])
                matrix[i, j] = float(row[key])
        im = ax.imshow(matrix, origin="lower", cmap="viridis", aspect="equal")
        ax.set_xticks(range(len(levels)), [f"{x:g}" for x in levels])
        ax.set_yticks(range(len(levels)), [f"{x:g}" for x in levels])
        ax.set_xlabel("Rear RC height (mm)")
        ax.set_ylabel("Front RC height (mm)")
        ax.set_title(label)
        fig.colorbar(im, ax=ax, shrink=0.8)
        lo, hi = np.nanmin(matrix), np.nanmax(matrix)
        for i in range(len(levels)):
            for j in range(len(levels)):
                value = matrix[i, j]
                color = "white" if not np.isfinite(value) or value < lo + 0.6 * (hi - lo) else "black"
                ax.text(
                    j, i, format(value, fmt) if np.isfinite(value) else "N/A", ha="center", va="center", color=color
                )
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
