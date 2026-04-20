from __future__ import annotations

SAM2_LABEL_PALETTE: tuple[tuple[int, int, int], ...] = (
    (94, 64, 157),   # purple
    (239, 68, 68),   # red
    (34, 197, 94),   # green
    (218, 112, 44),  # orange
    (14, 165, 233),  # sky
    (245, 158, 11),  # amber
    (236, 72, 153),  # pink
    (16, 185, 129),  # emerald
    (59, 130, 246),  # blue
    (168, 85, 247),  # violet
)


def sam2_palette_rgb(label: int) -> tuple[int, int, int]:
    """Return deterministic RGB color for a positive label id."""
    if label <= 0:
        return 0, 0, 0
    idx = (int(label) - 1) % len(SAM2_LABEL_PALETTE)
    return SAM2_LABEL_PALETTE[idx]
