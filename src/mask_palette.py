from __future__ import annotations


# Color-vision-deficiency-friendly palette (derived from Paul Tol's bright scheme).
_BASE_MASK_PALETTE_HEX: tuple[str, ...] = (
    "#4477AA",  # blue
    "#66CCEE",  # cyan
    "#228833",  # green
    "#CCBB44",  # yellow-olive
    "#EE6677",  # red
    "#AA3377",  # purple
    "#BBBBBB",  # gray
    "#0077BB",  # strong blue
    "#33BBEE",  # light blue
    "#009988",  # teal
    "#EE7733",  # orange
    "#CC3311",  # brick red
    "#EE3377",  # magenta
)


def get_mask_palette_hex(color_count: int) -> tuple[str, ...]:
    if color_count <= 0:
        raise ValueError("color_count must be a positive integer.")
    return _BASE_MASK_PALETTE_HEX[: min(color_count, len(_BASE_MASK_PALETTE_HEX))]


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
