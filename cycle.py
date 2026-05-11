#!/usr/bin/env python3
"""
perception-action cycle — fleet math in BOTH directions on LIVE PLATO data.

Perception:  sense the world (read tiles → constraint graph → emergence → field gaps)
Action:      change the world (identify gap → submit tile → field reconfigures)
Re-perceive: how did the field change?

Usage:
  python3 cycle.py --room forge            # Full cycle
  python3 cycle.py --room forge --steps 3  # Multiple action steps
  python3 cycle.py --room forge --perceive-only  # Observe only
"""

import argparse
import json
import math
import random
import sys
import time
import urllib.error
import urllib.request

# Import fleet_math. Adjust path if needed — the repo is expected at
# ~/superinstance/fleet-math-py/ (sibling of this script's parent).
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FLEET_MATH_PATHS = [
    os.path.join(_SCRIPT_DIR, "..", "fleet-math-py"),
    os.environ.get("FLEET_MATH_PATH", ""),
    "/home/ubuntu/superinstance/fleet-math-py",
]
for _p in _FLEET_MATH_PATHS:
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from fleet_math import (
        ConstraintGraph,
        betti_1,
        emergence_severity,
        detect_emergence,
        connected_components,
        is_rigid,
        rigid_margin,
        Field,
    )
except ImportError as e:
    print(f"ERROR: Cannot import fleet_math: {e}", file=sys.stderr)
    print("Make sure fleet-math-py is available (sibling dir or FLEET_MATH_PATH)", file=sys.stderr)
    sys.exit(1)

# ── config ──────────────────────────────────────────────────────────────────

PLATO_HOST = "http://localhost:8847"
JACCARD_THRESHOLD = 0.15  # similarity threshold for constraint edges
FIELD_GRID = 8             # grid resolution for gap detection
FIELD_DENSITY_THRESHOLD = 0.3  # gap sensitivity
EMBED_SEED = 42           # deterministic embedding

# ── PLATO I/O ───────────────────────────────────────────────────────────────


def plato_get(path):
    """Fetch a JSON resource from PLATO."""
    url = f"{PLATO_HOST}{path}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"  ⚠ HTTP {e.code} from {url}", file=sys.stderr)
        return None
    except (urllib.error.URLError, ConnectionError, OSError) as e:
        print(f"  ⚠ Connection error: {e}", file=sys.stderr)
        return None


def plato_post(payload):
    """Submit a tile to PLATO."""
    url = f"{PLATO_HOST}/submit"
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    try:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else "{}"
        return {"error": f"HTTP {e.code}", "body": body}
    except (urllib.error.URLError, ConnectionError, OSError) as e:
        return {"error": str(e)}


def fetch_tiles(room):
    """Fetch all tiles from a PLATO room. Returns list of tile dicts."""
    data = plato_get(f"/room/{room}")
    if data is None:
        return []
    # Normalize: top-level list, or dict with "tiles" key
    if isinstance(data, dict):
        for key in ("tiles", "data", "messages", "items"):
            if key in data and isinstance(data[key], list):
                return data[key]
    if isinstance(data, list):
        return data
    return []


# ── text processing ─────────────────────────────────────────────────────────


def extract_text(tile):
    """Extract searchable text from a tile of any shape."""
    if isinstance(tile, str):
        return tile
    if isinstance(tile, dict):
        # Try common keys in priority order
        for key in ("question", "answer", "content", "text", "message", "title", "summary", "body"):
            val = tile.get(key)
            if val and isinstance(val, str):
                return val
        # Fallback: concat everything
        return " ".join(str(v) for v in tile.values() if isinstance(v, (str, int, float)))
    return str(tile)


def tokenize(text):
    """Simple word tokenizer for Jaccard."""
    tokens = []
    buf = []
    for ch in text.lower():
        if ch.isalpha() or ch == "'":
            buf.append(ch)
        else:
            if buf:
                w = "".join(buf)
                if len(w) > 1:
                    tokens.append(w)
                buf = []
    if buf:
        w = "".join(buf)
        if len(w) > 1:
            tokens.append(w)
    return tokens


def jaccard_similarity(tokens_a, tokens_b):
    """Jaccard index between two token sets."""
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


# ── embedding (pure-Python MDS) ─────────────────────────────────────────────


def spectral_embedding(tokens_list):
    """Embed token arrays into 2D via classical MDS.

    Returns list of (x, y) coordinates, one per tile.
    Uses only stdlib — power iteration on the double-centered
    similarity matrix.
    """
    n = len(tokens_list)
    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0)]

    # Build Jaccard similarity matrix
    S = [[0.0] * n for _ in range(n)]
    for i in range(n):
        S[i][i] = 1.0  # self-similarity
        for j in range(i + 1, n):
            s = jaccard_similarity(tokens_list[i], tokens_list[j])
            S[i][j] = s
            S[j][i] = s

    # Convert to distance: D_ij = 1 - S_ij
    D = [[1.0 - S[i][j] for j in range(n)] for i in range(n)]

    # Double-centering: B = -1/2 * J * D^2 * J
    # where J = I - (1/n) * 11^T
    # First compute D_squared
    D2 = [[D[i][j] * D[i][j] for j in range(n)] for i in range(n)]

    # Row means, col means, grand mean of D2
    row_means = [sum(row) / n for row in D2]
    col_means = [sum(D2[i][j] for i in range(n)) / n for j in range(n)]
    grand_mean = sum(row_means) / n

    # B_ij = -0.5 * (D2_ij - row_mean_i - col_mean_j + grand_mean)
    B = [[-0.5 * (D2[i][j] - row_means[i] - col_means[j] + grand_mean) for j in range(n)] for i in range(n)]

    # Power iteration for top 2 eigenvectors of B
    def power_iteration(B, num_components=2, max_iter=100):
        n = len(B)
        vecs = []
        vals = []

        # Deflation: iteratively find top eigenvector, then subtract
        # its contribution
        B_work = [row[:] for row in B]  # copy

        for _ in range(num_components):
            # Initialize random vector
            v = [random.uniform(-1, 1) for _ in range(n)]
            v_norm = math.sqrt(sum(x * x for x in v))
            v = [x / v_norm for x in v]

            prev_lambda = 0.0
            for _ in range(max_iter):
                # B * v
                Bv = [sum(B_work[i][j] * v[j] for j in range(n)) for i in range(n)]

                # Rayleigh quotient: lambda = v^T * B * v / (v^T * v)
                # Since v is unit: lambda = sum(v_i * Bv_i)
                lam = sum(v[i] * Bv[i] for i in range(n))

                # Normalize
                norm = math.sqrt(sum(x * x for x in Bv))
                if norm < 1e-15:
                    break
                v_next = [x / norm for x in Bv]

                # Check convergence
                diff = math.sqrt(sum((v_next[i] - v[i]) ** 2 for i in range(n)))
                v = v_next
                if diff < 1e-10 and abs(lam - prev_lambda) < 1e-10:
                    break
                prev_lambda = lam

            vecs.append(v)
            vals.append(prev_lambda)

            # Deflate: B -= lambda * v * v^T
            for i in range(n):
                for j in range(n):
                    B_work[i][j] -= prev_lambda * v[i] * v[j]

        return vals, vecs

    random.seed(EMBED_SEED)
    vals, vecs = power_iteration(B, num_components=2)

    # Scale by sqrt(eigenvalue)
    result = []
    for i in range(n):
        x = math.sqrt(abs(vals[0])) * vecs[0][i] if len(vecs) > 0 else 0.0
        y = math.sqrt(abs(vals[1])) * vecs[1][i] if len(vecs) > 1 else 0.0
        result.append((x, y))

    return result


# ── perception ──────────────────────────────────────────────────────────────


def perceive(room):
    """Run perception on a PLATO room.

    Returns dict with:
      - tiles: list of tile dicts
      - texts: extracted text per tile
      - tokens: tokenized text per tile
      - hashes: tile _hash values
      - graph: ConstraintGraph instance
      - betti_1, v_minus_2, epsilon, emergence, rigid, rigid_margin
      - consensus, violations
      - coords: 2D embedding per tile
      - field: Field instance
      - gaps: list of gap cells
      - coverage: field coverage percentage
    """
    tiles = fetch_tiles(room)
    if not tiles:
        print(f"\n  ⚠ No tiles found in room '{room}' (or PLATO unreachable)")
        return None

    texts = [extract_text(t) for t in tiles]
    tokens = [tokenize(t) for t in texts]
    hashes = [t.get("_hash", f"tile_{i}") for i, t in enumerate(tiles)]

    V = len(tiles)

    # ── Build constraint graph ──
    graph = ConstraintGraph()
    for i in range(V):
        graph.add_node(hashes[i])
    for i in range(V):
        for j in range(i + 1, V):
            sim = jaccard_similarity(tokens[i], tokens[j])
            if sim > JACCARD_THRESHOLD:
                graph.add_edge(hashes[i], hashes[j], sim)

    E = len(graph.edges)
    C = connected_components(graph)
    b1 = betti_1(graph)
    v_minus_2 = max(V - 2, 1)
    eps = emergence_severity(graph)
    emergent = detect_emergence(graph)
    consensus, violations = graph.check_consensus()
    rigid = is_rigid(graph)
    margin = rigid_margin(graph)

    # ── 2D embedding ──
    coords = spectral_embedding(tokens)

    # ── Build field ──
    field = Field(power=2.0)
    for i in range(V):
        x, y = coords[i]
        # Use confidence as weight; default 0.5
        weight = tiles[i].get("confidence", 0.5)
        field.embed(hashes[i], x, y, weight)

    # ── Detect gaps ──
    gaps_raw = field.gaps(grid_size=FIELD_GRID, density_threshold=FIELD_DENSITY_THRESHOLD)

    # ── Coverage estimation ──
    total_cells = FIELD_GRID * FIELD_GRID
    occupied_cells = total_cells - len(gaps_raw)
    coverage = (occupied_cells / total_cells) * 100 if total_cells > 0 else 0.0

    return {
        "tiles": tiles,
        "texts": texts,
        "tokens": tokens,
        "hashes": hashes,
        "V": V,
        "E": E,
        "C": C,
        "graph": graph,
        "betti_1": b1,
        "v_minus_2": v_minus_2,
        "epsilon": eps,
        "emergent": emergent,
        "consensus": consensus,
        "violations": violations,
        "rigid": rigid,
        "rigid_margin": margin,
        "coords": coords,
        "field": field,
        "gaps": gaps_raw,
        "coverage": coverage,
    }


def report_perception(result):
    """Print a formatted perception report."""
    if result is None:
        return

    emoji = "🚨" if result["emergent"] else ("⚠️" if result["epsilon"] > 0.3 else "✅")
    status = "EMERGENT" if result["emergent"] else "stable"
    if not result["emergent"] and result["epsilon"] > 0.3:
        status = "approaching"
    cons = "✓" if result["consensus"] else "✗"
    rig = "✓" if result["rigid"] else "✗"

    print(f"\n  ┌─ Perception ─────────────────────────────────────")
    print(f"  │ Room:          {result['tiles'][0].get('domain', 'unknown') if result['tiles'] else 'unknown'}")
    print(f"  │ Tiles:         {result['V']}")
    print(f"  │ Edges:         {result['E']}")
    print(f"  │ Components:    {result['C']}")
    print(f"  │ β₁:            {result['betti_1']:.1f}, V-2: {result['v_minus_2']}, ε: {result['epsilon']:.2f} {emoji} {status}")
    print(f"  │ Consensus:     {cons} (holonomy={result['violations'][0][1]:.3f} if violations exist)")
    if result['violations']:
        for i, (cycle, h) in enumerate(result['violations'][:2]):
            print(f"  │   Cycle {i+1}: {len(cycle)} nodes, product={h:.3f}")
    print(f"  │ Laman rigid:   {rig}")
    print(f"  │ Field coverage: {result['coverage']:.1f}%")

    # Show top gaps with content hints
    gaps = result["gaps"]
    if gaps:
        # Sort gaps by count (lowest = sparsest regions)
        gaps_sorted = sorted(gaps, key=lambda g: g["count"])
        print(f"  │ Top gaps:")
        for g in gaps_sorted[:3]:
            # Find nearest tile's question as hint
            nearest_question = _nearest_tile_question(g, result)
            print(f"  │   (x={g['x']:.2f}, y={g['y']:.2f}, density={g['count']}) \"{nearest_question[:60]}...\"")
    else:
        print(f"  │ Gaps:          None (field is saturated)")

    print(f"  └──────────────────────────────────────────────────")


def _nearest_tile_question(gap, result):
    """Find the question of the tile nearest to a gap coordinate."""
    gx, gy = gap["x"], gap["y"]
    best_dist = float("inf")
    best_idx = 0
    for i, (x, y) in enumerate(result["coords"]):
        d = math.hypot(x - gx, y - gy)
        if d < best_dist:
            best_dist = d
            best_idx = i
    # Extract question
    t = result["tiles"][best_idx]
    q = extract_text(t)
    return q[:60]


def _nearest_tile_info(gap, result):
    """Return (question, answer) of the nearest tile to a gap."""
    gx, gy = gap["x"], gap["y"]
    best_dist = float("inf")
    best_idx = 0
    for i, (x, y) in enumerate(result["coords"]):
        d = math.hypot(x - gx, y - gy)
        if d < best_dist:
            best_dist = d
            best_idx = i
    t = result["tiles"][best_idx]
    q = t.get("question", extract_text(t))
    a = t.get("answer", "")
    return q, a


# ── action ──────────────────────────────────────────────────────────────────


def act(result, room):
    """Fill the largest gap with a new tile.

    Generates content by extending the nearest tile's topic into the
    gap region.
    """
    gaps = result["gaps"]
    if not gaps:
        print("\n  ⚠ No gaps to fill. Field is saturated.")
        return None

    # Pick the sparsest gap (lowest count)
    gaps_sorted = sorted(gaps, key=lambda g: g["count"])
    gap = gaps_sorted[0]
    gx, gy = gap["x"], gap["y"]

    # Get nearest tile's content for context
    q, a = _nearest_tile_info(gap, result)

    # Generate a sensible tile filling the gap
    new_question = f"Exploring the gap at ({gx:.2f}, {gy:.2f}) — extending '{q[:50].rstrip()}'"
    new_answer = _generate_fill_content(gap, result, q, a)

    payload = {
        "domain": room,
        "question": new_question,
        "answer": new_answer,
        "source": "perception-action-demo",
        "confidence": 0.8,
        "tags": ["perception-action-demo", "field-fill"],
    }

    print(f"\n  ┌─ Action ──────────────────────────────────────────")
    print(f"  │ Filling gap at ({gx:.2f}, {gy:.2f})")
    print(f"  │ Question: {new_question[:70]}...")
    print(f"  │ Answer: {new_answer[:70]}...")

    response = plato_post(payload)

    if "status" in response and response["status"] == "accepted":
        tile_hash = response.get("tile_hash", "???")
        print(f"  │ ✅ Tile accepted. Hash: {tile_hash}")
        print(f"  │ Room tile count: {response.get('room_tile_count', '?')}")
    elif "error" in response:
        print(f"  │ ❌ Submit failed: {response['error']}")
    else:
        print(f"  │ ❓ Unexpected response: {json.dumps(response, indent=2)[:100]}")
    print(f"  └──────────────────────────────────────────────────")

    return response


def _generate_fill_content(gap, result, q, a):
    """Generate sensible content for a gap fill tile.

    Produces a substantive answer by extending the knowledge around
    the gap into the uncovered region.
    """
    # Collect nearby tile texts for synthesis
    gx, gy = gap["x"], gap["y"]
    nearby = []
    for i, (x, y) in enumerate(result["coords"]):
        d = math.hypot(x - gx, y - gy)
        if d < 0.5 and i < len(result["tiles"]):
            t = result["tiles"][i]
            t_a = t.get("answer", extract_text(t))
            nearby.append(t_a)

    # Synthesize from what we know
    if nearby:
        # Pick a random nearby answer snippet and reframe it
        sample = random.choice(nearby)
        # Take first sentence
        first_sent = sample.split(". ")[0] if ". " in sample else sample[:100]
        return (f"Field exploration at ({gx:.2f}, {gy:.2f}). "
                f"Bridging from nearby knowledge: {first_sent}. "
                f"This region of the constraint space connects "
                f"{len(nearby)} known tiles into a continuous surface. "
                f"Further exploration will reveal how these concepts "
                f"interpolate at this coordinate.")
    else:
        return (f"New domain exploration at ({gx:.2f}, {gy:.2f}). "
                f"This region bridges adjacent constraint clusters. "
                f"Field density here is {gap['count']} tiles, suggesting "
                f"an opportunity for deeper knowledge embedding.")


# ── re-perception ───────────────────────────────────────────────────────────


def report_reperception(before, after):
    """Print the delta between two perception snapshots."""
    if before is None or after is None:
        return

    v_delta = after["V"] - before["V"]
    e_delta = after["E"] - before["E"]
    eps_delta = after["epsilon"] - before["epsilon"]
    cov_delta = after["coverage"] - before["coverage"]

    print(f"\n  ┌─ Re-perception (delta) ───────────────────────────")
    print(f"  │ Tiles:      {before['V']} → {after['V']} ({v_delta:+d})")
    print(f"  │ Edges:      {before['E']} → {after['E']} ({e_delta:+d})")
    print(f"  │ Coverage:   {before['coverage']:.1f}% → {after['coverage']:.1f}% ({cov_delta:+.1f}%)")
    print(f"  │ ε:          {before['epsilon']:.2f} → {after['epsilon']:.2f} ({eps_delta:+.2f})")
    if before["gaps"] and after["gaps"]:
        print(f"  │ Gaps:       {len(before['gaps'])} → {len(after['gaps'])}")
        # Check if new gaps opened
        new_gaps = len(after["gaps"]) - len(before["gaps"])
        if new_gaps > 0:
            print(f"  │ ⚠ New gaps opening — field is restructuring")
    elif not before["gaps"] and after["gaps"]:
        print(f"  │ Gaps:       none → {len(after['gaps'])} (field restructuring)")
    print(f"  └──────────────────────────────────────────────────")


# ── main ────────────────────────────────────────────────────────────────────


def main():
    global JACCARD_THRESHOLD, FIELD_GRID

    parser = argparse.ArgumentParser(
        description="Perception-action cycle — fleet math in both directions"
    )
    parser.add_argument("--room", default="forge", help="PLATO room to cycle on")
    parser.add_argument("--steps", type=int, default=1, help="Number of action steps")
    parser.add_argument("--perceive-only", action="store_true",
                        help="Perception only (no action, just observe)")
    parser.add_argument("--jaccard-threshold", type=float, default=0.15,
                        help="Jaccard threshold for edges (default 0.15)")
    parser.add_argument("--grid", type=int, default=8,
                        help="Field grid resolution (default 8)")
    args = parser.parse_args()

    JACCARD_THRESHOLD = args.jaccard_threshold
    FIELD_GRID = args.grid

    print(f"\n🔮 Perception-Action Cycle")
    print(f"   Room: {args.room}")
    print(f"   Steps: {'perceive-only' if args.perceive_only else args.steps}")
    print(f"   Jaccard threshold: {JACCARD_THRESHOLD}")
    print(f"   Field grid: {FIELD_GRID}×{FIELD_GRID}")
    print(f"   PLATO: {PLATO_HOST}")
    print(f"   FleetMath: {_FLEET_MATH_PATHS}")

    if args.perceive_only:
        # ── Perception only ──
        print(f"\n{'='*60}")
        print("  PERCEPTION (observe only)")
        print(f"{'='*60}")
        result = perceive(args.room)
        report_perception(result)
        print(f"\n{'='*60}")
        return

    # ── Cycle: perceive → act → re-perceive ──
    before = perceive(args.room)
    report_perception(before)

    for step in range(args.steps):
        print(f"\n{'='*60}")
        print(f"  CYCLE STEP {step + 1}/{args.steps}")
        print(f"{'='*60}")

        # Act
        if before and before["gaps"]:
            resp = act(before, args.room)
            # Small delay to let PLATO process
            time.sleep(0.5)
        else:
            print(f"\n  ⚠ No gaps to fill — field is saturated.")
            break

        # Re-perceive
        after = perceive(args.room)
        report_reperception(before, after)
        before = after

    print(f"\n{'='*60}")
    print("  CYCLE COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
