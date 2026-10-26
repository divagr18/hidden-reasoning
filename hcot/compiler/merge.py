"""Weight-merge helpers for the planted model (the subset of chipformer/q35/merge.py that
hcot.plant uses)."""

import re

IN_COLS = re.compile(
    r"(in_proj_(qkv|z|a|b)|q_proj|k_proj|v_proj|gate_proj|up_proj)\.weight$|embed_tokens\.weight$"
)
OUT_ROWS = re.compile(r"(out_proj|o_proj|down_proj)\.weight$|visual\.merger\.linear_fc2\.(weight|bias)$")


def pick_hidden(base: int, n_chip: int) -> tuple[int, int]:
    """Smallest (hidden, m) with hidden = m * 4^k >= base + m + n_chip, a multiple of 128, and
    3 <= m <= 64 ballast dims (m * B^2 / hidden must be a power of 4)."""
    best = None
    for m in range(3, 65):
        k = 0
        while m * 4**k < base + m + n_chip or (m * 4**k) % 128:
            k += 1
        h = m * 4**k
        if best is None or h < best[0] or (h == best[0] and m > best[1]):
            best = (h, m)
    return best
