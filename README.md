# perception-action — ARCHIVED

> **This repository is no longer actively maintained.**
> Superseded by: **[github.com/SuperInstance/fleet-math-c](https://github.com/SuperInstance/fleet-math-c)**
> Reason: SIMD perception-action loops now in fleet-math-c

---

# Perception-Action Cycle

Fleet math in BOTH directions on LIVE [PLATO](https://github.com/SuperInstance/plato) data.

```
Perception:  sense the world (read tiles → constraint graph → emergence → field gaps)
Action:      change the world (identify gap → submit tile → field reconfigures)
Re-perceive: how did the field change?
```

## Quick Start

```bash
# Full cycle on the forge room
python3 cycle.py --room forge

# Multiple action steps
python3 cycle.py --room forge --steps 3

# Perception only (no action, just observe)
python3 cycle.py --room forge --perceive-only
```

## What It Tests

**Perception direction:**
- Build a constraint graph from real PLATO tiles using Jaccard similarity
- Compute ZHC consensus (zero holonomy), H1 emergence (β₁, ε), Laman rigidity
- Embed tiles into 2D via classical MDS (pure Python, no numpy)
- Find gaps in the knowledge field

**Action direction:**
- Given the largest gap, generate a sensible tile that bridges toward it
- Submit the tile to PLATO via `/submit`
- Verify PLATO accepts the tile (status: `accepted`)
- Re-perceive to show how the field changed

**Re-perception:**
- Tile count changed?
- Edges changed? (new connections formed)
- Field coverage changed?
- Emergence severity (ε) shifted?
- New gaps opening?

## Architecture

```
cycle.py
├── PLATO I/O           — GET /room/{room}, POST /submit
├── Text processing     — extract_text, tokenize, jaccard_similarity
├── Spectral embedding  — pure-Python classical MDS (power iteration)
├── Perception          — build graph + field, compute metrics, find gaps
├── Action              — fill largest gap with new tile
└── Re-perception       — delta report
```

Uses [`fleet-math-py`](https://github.com/SuperInstance/fleet-math-py) for constraint graphs, emergence metrics, and Laman rigidity.

## Requirements

- Python 3.8+
- PLATO server running on `http://localhost:8847`
- `fleet-math-py` as a sibling directory or at `FLEET_MATH_PATH`

Zero external dependencies beyond stdlib and `fleet_math`.
