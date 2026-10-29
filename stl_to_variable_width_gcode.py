#!/usr/bin/env python3
"""Insert alternating variable-width paths into a PrusaSlicer ASCII G-code print.

The STL has no centerline metadata. This script projects the largest mesh
components onto a chosen plane, skeletonizes the silhouette, and follows the
resulting medial path. It inserts the path into selected layers of a host
G-code file, alternating direction on each layer. Per-layer height and active
extrusion-flow settings come from the host G-code.

Requires NumPy. Crossing review runs in a local browser page.
"""

from __future__ import annotations

import heapq
import html
import json
import math
import re
import struct
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Callable, Sequence

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment-dependent message
    raise SystemExit("This script requires NumPy. Install it with: python -m pip install numpy") from exc


# ------------------------------- User inputs --------------------------------

INPUT_STL = Path("One+Line+Frau.stl")
INPUT_GCODE_FILE = Path("One+Line+Frau_4realz_0.6n_0.25mm_PETG_MINIIS_5m.gcode")
OUTPUT_GCODE_FILE = Path("One+Line+Frau_variable_width.gcode")

# Layer numbers are 1-based, counting the first ;LAYER_CHANGE as layer 1.
# Set START_LAYER to the layer where the route should begin above the pedestal.
START_LAYER = 1
LAYER_COUNT = 4

MIN_EXTRUSION_WIDTH_MM: float | None = None  # None: use PrusaSlicer's external perimeter width.
MAX_EXTRUSION_WIDTH_MM = 1.20
MAX_FLOW_RATE_MM3_S: float | None = None  # None: use the lowest positive PrusaSlicer flow cap.
FLOW_SAFETY_FACTOR: float | None = None  # Custom G-code comment only; otherwise use the fallback below.
FLOW_SAFETY_FACTOR_FALLBACK = 0.98  # No standard PrusaSlicer setting corresponds to this extra margin.

# Additional printer and path settings.
FILAMENT_DIAMETER_MM_OVERRIDE: float | None = None  # Otherwise read a G-code comment; fall back to 1.75 mm.
FILAMENT_DIAMETER_FALLBACK_MM = 1.75
PRINT_SPEED_MM_S: float | None = None  # None: use external_perimeter_speed.
TRAVEL_SPEED_MM_S: float | None = None  # None: use travel_speed.
# Maximum centerline gap the script may connect. This is separate from the
# bead-width limits: connectors are printed at a capped width. Lower it if you
# want the script to reject uncertain joins rather than bridge them.
MAX_BRIDGE_GAP_MM = 8.0
# Raw raster skeletons can split one physical crossing into two nearby
# degree-3 junctions. Collapse odd junction pairs within this span so the
# review window presents the crossing as an even-arm junction.
MAX_CROSSING_CLUSTER_SPAN_MM = 8.0
RASTER_PIXELS_PER_MM = 5.0
PATH_SAMPLE_SPACING_MM = 0.50
PATH_SIMPLIFY_TOLERANCE_MM = 0.20
WIDTH_SIMPLIFY_TOLERANCE_MM = 0.08
SPUR_PRUNE_LENGTH_MM = 0.60
VERTEX_WELD_TOLERANCE_MM = 0.001
IGNORE_COMPONENTS_BELOW_TRIANGLES = 100
REVIEW_CROSSINGS_GRAPHICALLY = True
ARM_PREVIEW_LENGTH_MM = 8.0

# The line art is viewed along STL X, so STL Y/Z become printer X/Y. Change
# these axes if the model uses a different drawing plane. The coordinates are
# centered at the configured target point; adjust this to align with the
# replacement pedestal in the host G-code.
PROJECTION_AXES = (1, 2)  # horizontal, vertical; (0, 1) means view along Z
OUTPUT_CENTER_X_MM = 110.0
OUTPUT_CENTER_Y_MM = 110.0
OUTPUT_SCALE = 1.0

# The inserted path uses absolute XY and relative E. The script checks these
# modes at each insertion point and compensates for the host's active M221 flow.


# ------------------------------ STL handling --------------------------------

Point3 = tuple[float, float, float]
Triangle = tuple[Point3, Point3, Point3]


def load_stl(path: Path) -> list[Triangle]:
    data = path.read_bytes()
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        if 84 + count * 50 == len(data):
            triangles: list[Triangle] = []
            for i in range(count):
                values = struct.unpack_from("<12f", data, 84 + i * 50)
                triangles.append((
                    (values[3], values[4], values[5]),
                    (values[6], values[7], values[8]),
                    (values[9], values[10], values[11]),
                ))
            return triangles

    # ASCII STL fallback.
    try:
        lines = data.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is neither a supported binary nor ASCII STL") from exc
    vertices: list[Point3] = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not vertices or len(vertices) % 3:
        raise ValueError(f"Could not parse triangles from ASCII STL: {path}")
    return [tuple(vertices[i:i + 3]) for i in range(0, len(vertices), 3)]  # type: ignore[list-item]


def split_face_components(
    triangles: Sequence[Triangle], weld_tolerance: float
) -> list[list[int]]:
    """Group facets by shared/welded vertices, allowing pedestal filtering."""
    vertex_index: dict[tuple[int, int, int], int] = {}
    vertex_faces: list[list[int]] = []
    face_vertices: list[tuple[int, int, int]] = []
    inv_tol = 1.0 / max(weld_tolerance, 1e-9)

    for face_id, triangle in enumerate(triangles):
        ids: list[int] = []
        for point in triangle:
            key = tuple(int(round(value * inv_tol)) for value in point)
            vertex_id = vertex_index.get(key)
            if vertex_id is None:
                vertex_id = len(vertex_faces)
                vertex_index[key] = vertex_id
                vertex_faces.append([])
            vertex_faces[vertex_id].append(face_id)
            ids.append(vertex_id)
        face_vertices.append((ids[0], ids[1], ids[2]))

    seen = bytearray(len(triangles))
    components: list[list[int]] = []
    for seed in range(len(triangles)):
        if seen[seed]:
            continue
        seen[seed] = 1
        stack = [seed]
        component: list[int] = []
        while stack:
            face_id = stack.pop()
            component.append(face_id)
            for vertex_id in face_vertices[face_id]:
                for neighbor in vertex_faces[vertex_id]:
                    if not seen[neighbor]:
                        seen[neighbor] = 1
                        stack.append(neighbor)
        components.append(component)
    return components


# ----------------------------- Silhouette mask ------------------------------

def project_bounds(
    triangles: Sequence[Triangle], axes: tuple[int, int]
) -> tuple[float, float, float, float]:
    u_values = [point[axes[0]] for triangle in triangles for point in triangle]
    v_values = [point[axes[1]] for triangle in triangles for point in triangle]
    return min(u_values), max(u_values), min(v_values), max(v_values)


def rasterize_projected_triangles(
    triangles: Sequence[Triangle], axes: tuple[int, int], pixels_per_mm: float
) -> tuple[np.ndarray, tuple[float, float, float, float], int]:
    """Rasterize the projected mesh surface as a union of filled triangles."""
    u_min, u_max, v_min, v_max = project_bounds(triangles, axes)
    padding = max(3, int(math.ceil(2.0 * pixels_per_mm)))
    width = int(math.ceil((u_max - u_min) * pixels_per_mm)) + 2 * padding + 1
    height = int(math.ceil((v_max - v_min) * pixels_per_mm)) + 2 * padding + 1
    mask = np.zeros((height, width), dtype=np.bool_)

    for triangle in triangles:
        projected = [(point[axes[0]], point[axes[1]]) for point in triangle]
        row_coords = [(v_max - v) * pixels_per_mm + padding for _, v in projected]
        row_start = max(0, int(math.ceil(min(row_coords) - 0.5)))
        row_end = min(height - 1, int(math.floor(max(row_coords) - 0.5)))
        if row_start > row_end:
            continue

        for row in range(row_start, row_end + 1):
            scan_v = v_max + (padding - (row + 0.5)) / pixels_per_mm
            crossings: list[float] = []
            for i in range(3):
                u0, v0 = projected[i]
                u1, v1 = projected[(i + 1) % 3]
                # Half-open edge rule avoids counting a shared vertex twice.
                if (v0 <= scan_v < v1) or (v1 <= scan_v < v0):
                    crossings.append(u0 + (scan_v - v0) * (u1 - u0) / (v1 - v0))
            if len(crossings) < 2:
                continue
            c0 = max(0, int(math.ceil((min(crossings) - u_min) * pixels_per_mm + padding - 0.5)))
            c1 = min(width - 1, int(math.floor((max(crossings) - u_min) * pixels_per_mm + padding - 0.5)))
            if c0 <= c1:
                mask[row, c0:c1 + 1] = True
    return mask, (u_min, u_max, v_min, v_max), padding


def chamfer_distance(mask: np.ndarray) -> np.ndarray:
    """Approximate Euclidean distance to the silhouette boundary in pixels."""
    height, width = mask.shape
    inf = np.float32(1e9)
    diagonal = np.float32(math.sqrt(2.0))
    distance = np.where(mask, inf, 0.0).astype(np.float32)
    columns = np.arange(width, dtype=np.float32)

    for row_id in range(height):
        row = distance[row_id]
        if row_id:
            previous = distance[row_id - 1]
            left_diag = np.empty_like(previous)
            right_diag = np.empty_like(previous)
            left_diag[0] = inf
            left_diag[1:] = previous[:-1]
            right_diag[-1] = inf
            right_diag[:-1] = previous[1:]
            np.minimum(row, previous + 1.0, out=row)
            np.minimum(row, left_diag + diagonal, out=row)
            np.minimum(row, right_diag + diagonal, out=row)
        np.minimum(row, np.minimum.accumulate(row - columns) + columns, out=row)

    for row_id in range(height - 1, -1, -1):
        row = distance[row_id]
        if row_id + 1 < height:
            following = distance[row_id + 1]
            left_diag = np.empty_like(following)
            right_diag = np.empty_like(following)
            left_diag[0] = inf
            left_diag[1:] = following[:-1]
            right_diag[-1] = inf
            right_diag[:-1] = following[1:]
            np.minimum(row, following + 1.0, out=row)
            np.minimum(row, left_diag + diagonal, out=row)
            np.minimum(row, right_diag + diagonal, out=row)
        suffix = np.minimum.accumulate((row + columns)[::-1])[::-1] - columns
        np.minimum(row, suffix, out=row)
    return distance


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning, vectorized with NumPy."""
    image = mask.copy()
    for _ in range(1000):
        removed_any = False
        for phase in (0, 1):
            padded = np.pad(image, 1, mode="constant", constant_values=False)
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            neighbors = (p2, p3, p4, p5, p6, p7, p8, p9)
            neighbor_count = sum(neighbor.astype(np.uint8) for neighbor in neighbors)
            transitions = sum(
                ((~neighbors[i]) & neighbors[(i + 1) % 8]).astype(np.uint8)
                for i in range(8)
            )
            if phase == 0:
                condition = (~(p2 & p4 & p6)) & (~(p4 & p6 & p8))
            else:
                condition = (~(p2 & p4 & p8)) & (~(p2 & p6 & p8))
            remove = image & (neighbor_count >= 2) & (neighbor_count <= 6) & (transitions == 1) & condition
            if np.any(remove):
                image[remove] = False
                removed_any = True
        if not removed_any:
            break
    return image


# -------------------------- Skeleton graph/path -----------------------------

def build_graph(skeleton: np.ndarray) -> tuple[list[tuple[int, int]], list[list[int]]]:
    coordinates = [tuple(map(int, point)) for point in np.argwhere(skeleton)]
    if not coordinates:
        raise ValueError("The projected mesh produced an empty skeleton")
    index = np.full(skeleton.shape, -1, dtype=np.int32)
    for node_id, (row, col) in enumerate(coordinates):
        index[row, col] = node_id
    height, width = skeleton.shape
    adjacency: list[list[int]] = [[] for _ in coordinates]
    for node_id, (row, col) in enumerate(coordinates):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = row + dr, col + dc
                if rr < 0 or cc < 0 or rr >= height or cc >= width or not skeleton[rr, cc]:
                    continue
                # Suppress diagonal shortcuts around an existing orthogonal
                # connection; this keeps the 8-connected skeleton a thin graph.
                if dr and dc and (skeleton[row, cc] or skeleton[rr, col]):
                    continue
                neighbor = int(index[rr, cc])
                if neighbor >= 0:
                    adjacency[node_id].append(neighbor)
    return coordinates, adjacency


def collapse_odd_junction_pairs(
    coordinates: Sequence[tuple[int, int]],
    adjacency: Sequence[Sequence[int]],
    max_span_px: float,
) -> tuple[list[tuple[int, int]], list[list[int]], int]:
    """Merge nearby odd branch nodes into even-arm crossing regions.

    Skeletonizing a rasterized crossing can leave two degree-3 pixels joined
    by a short chain. Contracting the whole chain turns that artifact into one
    degree-4 crossing node, which the reviewer can pair unambiguously.
    """
    degrees = [len(neighbors) for neighbors in adjacency]
    odd_junctions = {node for node, degree in enumerate(degrees) if degree >= 3 and degree % 2}
    if not odd_junctions:
        return list(coordinates), [list(neighbors) for neighbors in adjacency], 0

    candidate_paths: dict[tuple[int, int], tuple[float, list[int]]] = {}
    for start in sorted(odd_junctions):
        distances = {start: 0.0}
        previous: dict[int, int] = {}
        queue = [(0.0, start)]
        while queue:
            distance_px, node = heapq.heappop(queue)
            if distance_px > distances[node] or distance_px > max_span_px:
                continue
            for neighbor in adjacency[node]:
                step = math.dist(coordinates[node], coordinates[neighbor])
                total = distance_px + step
                if total > max_span_px:
                    continue
                neighbor_degree = degrees[neighbor]
                if neighbor_degree >= 3:
                    if neighbor == start or neighbor not in odd_junctions:
                        continue
                    path = [node, neighbor]
                    while path[0] != start:
                        path.insert(0, previous[path[0]])
                    first, second = start, neighbor
                    if first > second:
                        first, second = second, first
                        path.reverse()
                    key = (first, second)
                    old = candidate_paths.get(key)
                    if old is None or total < old[0]:
                        candidate_paths[key] = (total, path)
                    continue
                if neighbor_degree != 2:
                    continue
                if total < distances.get(neighbor, math.inf):
                    distances[neighbor] = total
                    previous[neighbor] = node
                    heapq.heappush(queue, (total, neighbor))

    candidates = sorted(
        (distance_px, first, second, path)
        for (first, second), (distance_px, path) in candidate_paths.items()
    )
    used_nodes: set[int] = set()
    merge_paths: list[list[int]] = []
    for _distance_px, first, second, path in candidates:
        if first in used_nodes or second in used_nodes or any(node in used_nodes for node in path):
            continue
        used_nodes.update(path)
        merge_paths.append(path)

    unmatched = odd_junctions - used_nodes
    if unmatched:
        examples = ", ".join(str(coordinates[node]) for node in sorted(unmatched)[:6])
        raise ValueError(
            f"Found {len(unmatched)} odd-arm junction(s) that cannot be grouped into even-arm crossings "
            "within the configured MAX_CROSSING_CLUSTER_SPAN_MM. "
            f"Example raster coordinates: {examples}. Increase that setting or inspect the STL."
        )

    representative_for: dict[int, int] = {}
    for path in merge_paths:
        representative = path[len(path) // 2]
        for node in path:
            representative_for[node] = representative

    roots = sorted({representative_for.get(node, node) for node in range(len(coordinates))})
    compact_index = {root: index for index, root in enumerate(roots)}
    collapsed_coordinates = [coordinates[root] for root in roots]
    collapsed_edges: set[tuple[int, int]] = set()
    for node, neighbors in enumerate(adjacency):
        root_node = representative_for.get(node, node)
        for neighbor in neighbors:
            if neighbor <= node:
                continue
            root_neighbor = representative_for.get(neighbor, neighbor)
            if root_node == root_neighbor:
                continue
            first = compact_index[root_node]
            second = compact_index[root_neighbor]
            collapsed_edges.add((min(first, second), max(first, second)))

    collapsed_adjacency: list[list[int]] = [[] for _ in collapsed_coordinates]
    for first, second in sorted(collapsed_edges):
        collapsed_adjacency[first].append(second)
        collapsed_adjacency[second].append(first)

    remaining_odd = [
        node for node, neighbors in enumerate(collapsed_adjacency)
        if len(neighbors) >= 3 and len(neighbors) % 2
    ]
    if remaining_odd:
        raise ValueError(
            "Could not normalize all raster junctions into even-arm crossings; "
            f"{len(remaining_odd)} odd junction(s) remain. Increase MAX_CROSSING_CLUSTER_SPAN_MM or inspect the STL."
        )
    return collapsed_coordinates, collapsed_adjacency, len(merge_paths)


def graph_components(adjacency: Sequence[Sequence[int]]) -> list[list[int]]:
    seen = bytearray(len(adjacency))
    result: list[list[int]] = []
    for seed in range(len(adjacency)):
        if seen[seed]:
            continue
        seen[seed] = 1
        stack = [seed]
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if not seen[neighbor]:
                    seen[neighbor] = 1
                    stack.append(neighbor)
        result.append(component)
    return result


def prune_short_spurs(skeleton: np.ndarray, prune_length_px: float) -> np.ndarray:
    """Remove short terminal branches that are raster/silhouette artifacts."""
    image = skeleton.copy()
    while True:
        coords, adjacency = build_graph(image)
        if len(coords) < 3:
            return image
        degrees = [len(neighbors) for neighbors in adjacency]
        remove: set[int] = set()
        for endpoint, degree in enumerate(degrees):
            if degree != 1:
                continue
            branch = [endpoint]
            previous = -1
            current = endpoint
            length = 0.0
            while True:
                candidates = [n for n in adjacency[current] if n != previous]
                if not candidates:
                    break
                following = candidates[0]
                a, b = coords[current], coords[following]
                length += math.hypot(a[0] - b[0], a[1] - b[1])
                if len(adjacency[current]) >= 3:
                    break
                previous, current = current, following
                if len(adjacency[current]) >= 3:
                    break
                branch.append(current)
                if len(adjacency[current]) != 2:
                    break
            if len(adjacency[current]) >= 3 and length < prune_length_px:
                remove.update(branch)
        if not remove:
            return image
        for node_id in remove:
            row, col = coords[node_id]
            image[row, col] = False


def endpoint_tangent(
    endpoint: int, coordinates: Sequence[tuple[int, int]], adjacency: Sequence[Sequence[int]]
) -> tuple[float, float]:
    neighbors = adjacency[endpoint]
    if not neighbors:
        return (0.0, 0.0)
    row, col = coordinates[endpoint]
    nr, nc = coordinates[neighbors[0]]
    dr, dc = row - nr, col - nc
    norm = math.hypot(dr, dc) or 1.0
    return dr / norm, dc / norm


def bresenham(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    row0, col0 = a
    row1, col1 = b
    drow, dcol = abs(row1 - row0), abs(col1 - col0)
    step_row = 1 if row0 < row1 else -1
    step_col = 1 if col0 < col1 else -1
    error = dcol - drow
    result: list[tuple[int, int]] = []
    while True:
        result.append((row0, col0))
        if row0 == row1 and col0 == col1:
            break
        twice = 2 * error
        if twice > -drow:
            error -= drow
            col0 += step_col
        if twice < dcol:
            error += dcol
            row0 += step_row
    return result


def trace_arm(
    junction: int,
    first_neighbor: int,
    coordinates: Sequence[tuple[int, int]],
    adjacency: Sequence[Sequence[int]],
    max_length_px: float,
) -> list[int]:
    """Trace one visible branch away from a junction for the review view."""
    path = [junction, first_neighbor]
    previous, current = junction, first_neighbor
    length = math.dist(coordinates[previous], coordinates[current])
    while length < max_length_px and len(adjacency[current]) == 2:
        options = [node for node in adjacency[current] if node != previous]
        if not options:
            break
        following = options[0]
        if following == junction:
            break
        length += math.dist(coordinates[current], coordinates[following])
        path.append(following)
        previous, current = current, following
    return path


def propose_crossing_pairs(
    junction: int,
    arm_starts: Sequence[int],
    coordinates: Sequence[tuple[int, int]],
) -> list[list[int]]:
    """Suggest a perfect pairing with nearly perpendicular crossing axes."""
    vectors = []
    row, col = coordinates[junction]
    for node in arm_starts:
        dr, dc = coordinates[node][0] - row, coordinates[node][1] - col
        length = math.hypot(dr, dc) or 1.0
        vectors.append((dr / length, dc / length))

    def matchings(remaining: tuple[int, ...]):
        if not remaining:
            yield []
            return
        first = remaining[0]
        for position in range(1, len(remaining)):
            second = remaining[position]
            rest = remaining[1:position] + remaining[position + 1:]
            for tail in matchings(rest):
                yield [(first, second), *tail]

    if not arm_starts or len(arm_starts) % 2:
        return []
    best_pairs: list[tuple[int, int]] = []
    best_score = math.inf
    for pairing in matchings(tuple(range(len(arm_starts)))):
        axes = []
        smoothness = 0.0
        for first, second in pairing:
            a, b = vectors[first], vectors[second]
            axis = (a[0] - b[0], a[1] - b[1])
            length = math.hypot(*axis) or 1.0
            axes.append((axis[0] / length, axis[1] / length))
            smoothness += 1.0 + a[0] * b[0] + a[1] * b[1]
        perpendicularity = sum(
            abs(a[0] * b[0] + a[1] * b[1])
            for i, a in enumerate(axes)
            for b in axes[i + 1:]
        )
        score = perpendicularity + 0.05 * smoothness
        if score < best_score:
            best_score, best_pairs = score, pairing
    return [list(pair) for pair in best_pairs]


def render_crossing_map(
    coordinates: Sequence[tuple[int, int]],
    adjacency: Sequence[Sequence[int]],
    junction_nodes: Sequence[int],
) -> str:
    rows = [row for row, _col in coordinates]
    cols = [col for _row, col in coordinates]
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    width, height = max(1, max_col - min_col), max(1, max_row - min_row)
    edges = "".join(
        f'<line x1="{coordinates[node][1]}" y1="{coordinates[node][0]}" '
        f'x2="{coordinates[neighbor][1]}" y2="{coordinates[neighbor][0]}"/>'
        for node, neighbors in enumerate(adjacency)
        for neighbor in neighbors
        if node < neighbor
    )
    radius = max(3.0, min(width, height) * 0.012)
    markers = "".join(
        f'<g class="crossing-marker" data-crossing-index="{index - 1}" role="button" '
        f'tabindex="0" aria-label="Review crossing {index}">'
        f'<circle cx="{coordinates[node][1]}" cy="{coordinates[node][0]}" r="{radius:.2f}"/>'
        f'<text x="{coordinates[node][1]}" y="{coordinates[node][0]}">{index}</text></g>'
        for index, node in enumerate(junction_nodes, start=1)
    )
    return (
        f'<svg viewBox="{min_col} {min_row} {width} {height}" role="img" aria-label="Line art overview">'
        f'<g class="model-lines">{edges}</g>{markers}</svg>'
    )


def review_crossing_pairings(
    skeleton: np.ndarray,
    coordinates: Sequence[tuple[int, int]],
    adjacency: Sequence[Sequence[int]],
    junction_nodes: Sequence[int],
    pixels_per_mm: float,
    validate_decisions: Callable[
        [dict[int, list[tuple[int, int]]]], tuple[str | None, str]
    ],
) -> dict[int, list[tuple[int, int]]]:
    """Offer the same visual crossing review in a local browser tab."""
    if not junction_nodes:
        return {}
    palette = (
        "#ff6b6b", "#4dabf7", "#69db7c", "#ffd43b", "#b197fc", "#38d9a9",
        "#f783ac", "#ffa94d", "#74c0fc", "#a9e34b", "#e599f7", "#66d9e8",
    )
    views: list[dict[str, object]] = []
    for crossing_index, junction in enumerate(junction_nodes, start=1):
        arms = [
            trace_arm(
                junction,
                neighbor,
                coordinates,
                adjacency,
                ARM_PREVIEW_LENGTH_MM * pixels_per_mm,
            )
            for neighbor in adjacency[junction]
        ]
        if len(arms) < 3:
            continue

        visible_nodes = [node for arm in arms for node in arm]
        rows = [coordinates[node][0] for node in visible_nodes]
        cols = [coordinates[node][1] for node in visible_nodes]
        margin = max(12.0, 2.5 * pixels_per_mm)
        min_row, max_row = min(rows) - margin, max(rows) + margin
        min_col, max_col = min(cols) - margin, max(cols) + margin
        canvas_w, canvas_h = 820.0, 570.0
        scale = min(
            (canvas_w - 48.0) / max(1.0, max_col - min_col),
            (canvas_h - 48.0) / max(1.0, max_row - min_row),
        )
        offset_x = (canvas_w - (max_col - min_col) * scale) * 0.5
        offset_y = (canvas_h - (max_row - min_row) * scale) * 0.5

        def screen(node: int) -> list[float]:
            row, col = coordinates[node]
            return [offset_x + (col - min_col) * scale, offset_y + (row - min_row) * scale]

        r0, r1 = max(0, int(min_row)), min(skeleton.shape[0] - 1, int(max_row))
        c0, c1 = max(0, int(min_col)), min(skeleton.shape[1] - 1, int(max_col))
        context_edges: list[list[list[float]]] = []
        for node, (row, col) in enumerate(coordinates):
            if not (r0 <= row <= r1 and c0 <= col <= c1):
                continue
            for neighbor in adjacency[node]:
                if neighbor <= node:
                    continue
                rr, cc = coordinates[neighbor]
                if r0 <= rr <= r1 and c0 <= cc <= c1:
                    context_edges.append([screen(node), screen(neighbor)])

        views.append({
            "crossing": crossing_index,
            "total": len(junction_nodes),
            "junction_node": junction,
            "arm_starts": [arm[1] for arm in arms],
            "junction": screen(junction),
            "context": context_edges,
            "arms": [
                {"points": [screen(node) for node in arm], "color": palette[i % len(palette)]}
                for i, arm in enumerate(arms)
            ],
        })

    state: dict[str, object] = {
        "index": 0,
        "show_overview": True,
        "done": False,
        "cancelled": False,
        "finished": False,
        "editing": False,
        "validation_error": None,
        "validation_overview": "",
        "validation_ok": False,
    }
    decisions: dict[int, list[tuple[int, int]]] = {}
    crossing_map = render_crossing_map(coordinates, adjacency, junction_nodes)

    class ReviewServer(HTTPServer):
        allow_reuse_address = True

    class ReviewHandler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            return

        def respond(self, body: str, status: int = 200) -> None:
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def page(self) -> str:
            if bool(state["cancelled"]):
                return "<h1>Crossing review cancelled</h1><p>No G-code was written.</p>"
            if bool(state["finished"]):
                state["done"] = True
                return "<h1>Route verified</h1><p>You can close this tab.</p>"
            if bool(state["show_overview"]):
                return CROSSING_OVERVIEW_HTML.replace("__OVERVIEW_SVG__", crossing_map)
            index = int(state["index"])
            if index >= len(views):
                error = state["validation_error"]
                safe_error = (
                    f'<p class="error" role="alert">{html.escape(str(error))}</p>'
                    if error
                    else ""
                )
                overview = str(state["validation_overview"] or "")
                action = (
                    '<button id="confirm">Confirm route and continue</button>'
                    if state["validation_ok"]
                    else '<p class="error-hint">Edit one or more crossings below, then the route will be checked again.</p>'
                )
                return (
                    FINAL_REVIEW_HTML.replace("__ERROR__", safe_error)
                    .replace("__OVERVIEW__", overview)
                    .replace("__ACTION__", action)
                )
            view = views[index]
            arm_index = {int(node): i for i, node in enumerate(view["arm_starts"])}
            junction_node = int(view["junction_node"])
            if junction_node in decisions:
                initial_pairs = [
                    [arm_index[first], arm_index[second]]
                    for first, second in decisions[junction_node]
                    if first in arm_index and second in arm_index
                ]
            else:
                initial_pairs = propose_crossing_pairs(
                    junction_node,
                    [int(node) for node in view["arm_starts"]],
                    coordinates,
                )
            view_json = json.dumps(
                {**view, "initial_pairs": initial_pairs}, separators=(",", ":")
            ).replace("<", "\\u003c")
            return BROWSER_REVIEW_HTML.replace("__VIEW_JSON__", view_json)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            self.respond(self.page())

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            if self.path == "/cancel":
                state["cancelled"] = True
                state["done"] = True
                self.respond("<h1>Crossing review cancelled</h1><p>No G-code was written.</p>")
                return
            if self.path not in {"/submit", "/edit", "/confirm", "/overview", "/return"}:
                self.respond("<h1>Not found</h1>", 404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                if self.path == "/edit":
                    edit_index = payload["index"]
                    if not isinstance(edit_index, int) or not 0 <= edit_index < len(views):
                        raise ValueError("Invalid crossing index")
                    state["index"] = edit_index
                    state["editing"] = True
                    state["validation_error"] = None
                    state["validation_overview"] = ""
                    state["validation_ok"] = False
                    state["show_overview"] = False
                    self.respond("<p>Opening crossing for editing…</p>")
                    return
                if self.path == "/return":
                    state["show_overview"] = False
                    self.respond("<p>Returning to crossing review…</p>")
                    return
                if self.path == "/confirm":
                    if not bool(state["validation_ok"]):
                        raise ValueError("Validate the route overview before confirming")
                    state["finished"] = True
                    self.respond("<p>Route confirmed. Finishing…</p>")
                    return
                submitted_pairs = payload["pairs"]
                view = views[int(state["index"])]
                arm_starts = view["arm_starts"]
                if not isinstance(arm_starts, list):
                    raise ValueError("Invalid arm data")
                pairs: list[tuple[int, int]] = []
                used_arms: set[int] = set()
                for pair in submitted_pairs:
                    if (
                        not isinstance(pair, list)
                        or len(pair) != 2
                        or any(not isinstance(arm_id, int) for arm_id in pair)
                        or any(arm_id < 0 or arm_id >= len(arm_starts) for arm_id in pair)
                        or pair[0] == pair[1]
                        or any(arm_id in used_arms for arm_id in pair)
                    ):
                        raise ValueError("Invalid arm pairing")
                    used_arms.update(pair)
                    pairs.append((int(arm_starts[pair[0]]), int(arm_starts[pair[1]])))
            except (ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError):
                self.respond("<h1>Invalid crossing selection</h1><p>Reload this page and try again.</p>", 400)
                return

            crossing_node = int(view["junction_node"])
            decisions[crossing_node] = pairs
            if self.path == "/overview":
                state["show_overview"] = True
                self.respond("<p>Opening crossing overview…</p>")
                return
            if bool(state["editing"]):
                state["index"] = len(views)
                state["editing"] = False
            else:
                state["index"] = int(state["index"]) + 1
            if int(state["index"]) >= len(views):
                error, overview = validate_decisions(decisions)
                state["validation_error"] = error
                state["validation_overview"] = overview
                state["validation_ok"] = error is None
            self.respond("<p>Selection saved. Loading the next crossing…</p>")

    with ReviewServer(("127.0.0.1", 0), ReviewHandler) as server:
        server.timeout = 0.5
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"Crossing review opened in a browser: {url}", file=sys.stderr, flush=True)
        try:
            webbrowser.open(url, new=2)
        except Exception:
            pass
        while not bool(state["done"]):
            server.handle_request()

    if bool(state["cancelled"]):
        raise SystemExit("Crossing review canceled; no G-code snippets were written.")
    return decisions


BROWSER_REVIEW_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Line-art crossing review</title>
<style>
body{margin:0;background:#111827;color:#f8fafc;font:16px system-ui,sans-serif}
main{max-width:920px;margin:0 auto;padding:18px}h1{font-size:21px;margin:0 0 8px}
p{color:#cbd5e1;margin:6px 0}svg{display:block;width:100%;height:auto;background:#111827;border:1px solid #334155;border-radius:8px;margin:14px 0}
button{font:inherit;padding:9px 15px;border:0;border-radius:5px;cursor:pointer;margin-right:8px}
#continue{background:#69db7c;color:#102018;float:right}#cancel{background:#475569;color:#fff}
.arm{fill:none;stroke-width:5;stroke-linecap:round;stroke-linejoin:round}.tip{fill:#111827;stroke-width:3;cursor:pointer}
.tip.selected{stroke:#fff;stroke-width:5}.label{fill:#fff;text-anchor:middle;dominant-baseline:central;font-weight:700;pointer-events:none}
.pair{stroke:#f8f9fa;stroke-width:2;stroke-dasharray:6 4}.context{stroke:#526071;stroke-width:2}.junction{fill:#fff;stroke:#111827;stroke-width:2}
</style></head><body><main>
<h1 id="heading"></h1><p>Click two colored arm tips to connect the line ends that continue through this crossing.</p>
<p>Suggested pairs favor line directions that cross near 90°. Adjust them if needed, then choose Continue. Unpaired arms use the route fallback.</p>
<svg id="drawing" viewBox="0 0 820 570" role="img" aria-label="Line crossing and selectable arm ends"></svg>
<button id="undo">Undo pair</button><button id="clear">Clear</button><button id="overview">Overview</button><button id="cancel">Cancel</button><button id="continue">Continue</button>
</main><script>
const view=__VIEW_JSON__, svg=document.getElementById('drawing');let pairs=view.initial_pairs??[],selected=[];
document.getElementById('heading').textContent=`Crossing ${view.crossing} of ${view.total} — ${view.arms.length} line ends`;
function el(name,attrs={}){const n=document.createElementNS('http://www.w3.org/2000/svg',name);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);return n;}
for(const edge of view.context)svg.append(el('line',{x1:edge[0][0],y1:edge[0][1],x2:edge[1][0],y2:edge[1][1],class:'context'}));
const tips=[];
view.arms.forEach((arm,i)=>{const p=arm.points;svg.append(el('polyline',{points:p.map(q=>q.join(',')).join(' '),stroke:arm.color,class:'arm'}));tips.push(p[p.length-1]);});
const pairLayer=el('g');svg.append(pairLayer);
function render(){pairLayer.replaceChildren();for(const pair of pairs){const a=tips[pair[0]],b=tips[pair[1]];pairLayer.append(el('line',{x1:a[0],y1:a[1],x2:b[0],y2:b[1],class:'pair'}));}
svg.querySelectorAll('.arm-tip').forEach(n=>n.remove());view.arms.forEach((arm,i)=>{const [x,y]=tips[i],g=el('g',{class:'arm-tip','data-arm':i});g.append(el('circle',{cx:x,cy:y,r:pairs.some(p=>p.includes(i))?9:12,fill:'#111827',stroke:arm.color,class:selected.includes(i)?'tip selected':'tip'}));const t=el('text',{x,y,class:'label'});t.textContent=String.fromCharCode(65+i);g.append(t);svg.append(g);});
const [x,y]=view.junction;svg.append(el('circle',{cx:x,cy:y,r:7,class:'junction'}));}
svg.addEventListener('click',event=>{const g=event.target.closest('[data-arm]');if(!g)return;const i=Number(g.dataset.arm);if(pairs.some(p=>p.includes(i)))return;if(selected.includes(i))selected=selected.filter(x=>x!==i);else selected.push(i);if(selected.length===2){pairs.push([...selected]);selected=[];}render();});
document.getElementById('undo').onclick=()=>{if(selected.length)selected=[];else pairs.pop();render();};
document.getElementById('clear').onclick=()=>{pairs=[];selected=[];render();};
async function send(path,body){for(const b of document.querySelectorAll('button'))b.disabled=true;const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});if(!r.ok){document.body.innerHTML='<main><h1>Could not submit crossing selection</h1><p>Reload the page and try again.</p></main>';return;}location.reload();}
document.getElementById('continue').onclick=()=>send('/submit',{pairs});document.getElementById('overview').onclick=()=>send('/overview',{pairs});document.getElementById('cancel').onclick=()=>send('/cancel',{});render();
</script></body></html>"""


CROSSING_OVERVIEW_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Crossing overview</title>
<style>body{margin:0;background:#111827;color:#f8fafc;font:16px system-ui,sans-serif}main{max-width:960px;margin:auto;padding:20px}
h1{font-size:22px}p{color:#cbd5e1}svg{width:100%;max-height:72vh;background:#0b1220;border:1px solid #334155;border-radius:8px}
.model-lines{stroke:#8492a6;stroke-width:1.5}.crossing-marker{cursor:pointer}.crossing-marker circle{fill:#fff;stroke:#111827;stroke-width:1.5}
.crossing-marker text{fill:#111827;text-anchor:middle;dominant-baseline:central;font:bold 8px sans-serif;pointer-events:none}
.crossing-marker:hover circle,.crossing-marker:focus circle{fill:#ffd43b;stroke:#fff;stroke-width:2.5}
button{font:inherit;padding:9px 14px;margin:8px 8px 0 0;border:0;border-radius:5px;cursor:pointer;background:#69db7c;color:#102018}</style></head>
<body><main><h1>Crossing overview</h1><p>Click a numbered crossing to review it, or resume the crossing sequence. Use Overview on any crossing page to return here.</p>
__OVERVIEW_SVG__<button id="resume">Start / resume crossing review</button><button id="cancel">Cancel</button></main>
<script>async function post(path,body){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});if(r.ok)location.reload();}
document.getElementById('resume').onclick=()=>post('/return',{});document.getElementById('cancel').onclick=()=>post('/cancel',{});
document.querySelectorAll('.crossing-marker').forEach(marker=>{const open=()=>post('/edit',{index:Number(marker.dataset.crossingIndex)});marker.onclick=open;marker.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();open();}};});</script></body></html>"""


FINAL_REVIEW_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Verify line-art route</title><style>
body{margin:0;background:#111827;color:#f8fafc;font:16px system-ui,sans-serif}
main{max-width:920px;margin:0 auto;padding:24px}h1{font-size:22px;margin:0 0 10px}
p{color:#cbd5e1;line-height:1.5}.error{padding:12px 14px;background:#7f1d1d;color:#fee2e2;border-radius:6px}
.error-hint{padding:10px 12px;background:#334155;border-radius:6px}
button{font:inherit;padding:9px 14px;border:0;border-radius:5px;cursor:pointer;margin:5px 7px 5px 0}
#confirm{background:#69db7c;color:#102018;font-weight:650}#cancel{background:#475569;color:#fff}
.crossing-marker{cursor:pointer}.crossing-marker:hover circle,.crossing-marker:focus circle{fill:#ffd43b;stroke:#fff;stroke-width:2.5}
svg{display:block;width:100%;max-height:560px;background:#0b1220;border:1px solid #334155;border-radius:7px;margin:12px 0}
.legend{display:flex;flex-wrap:wrap;gap:8px 18px}.legend span{display:flex;align-items:center;gap:7px;color:#cbd5e1}
.legend i{width:12px;height:12px;border-radius:50%;display:inline-block}
</style></head><body><main>
<h1>Review the complete route</h1>
__ERROR__
__OVERVIEW__
<p>Review the complete route before finishing. If it splits into separate trails, edit one or more crossings and the route will be checked again. No G-code is written until you confirm the overview.</p>
__ACTION__<button id="cancel">Cancel</button>
</main><script>
async function post(path,body){for(const b of document.querySelectorAll('button'))b.disabled=true;const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});if(!r.ok){document.body.innerHTML='<main><h1>Could not update crossing review</h1><p>Reload the page and try again.</p></main>';return;}location.reload();}
document.getElementById('confirm')?.addEventListener('click',()=>post('/confirm',{}));
document.getElementById('cancel').onclick=()=>post('/cancel',{});
document.querySelectorAll('.crossing-marker').forEach(marker=>{
const open=()=>post('/edit',{index:Number(marker.dataset.crossingIndex)});
marker.addEventListener('click',open);
marker.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();open();}});
});
</script></body></html>"""


def connect_nearby_components(
    skeleton: np.ndarray,
    distance: np.ndarray,
    pixels_per_mm: float,
    max_gap_mm: float,
    min_width_mm: float,
    max_width_mm: float,
) -> tuple[np.ndarray, dict[tuple[int, int], float], int]:
    """Join nearby open ends with short printed bridges; reject distant gaps."""
    image = skeleton.copy()
    bridge_widths: dict[tuple[int, int], float] = {}
    bridges = 0
    while True:
        coordinates, adjacency = build_graph(image)
        components = graph_components(adjacency)
        if len(components) <= 1:
            return image, bridge_widths, bridges

        component_endpoints: list[list[int]] = []
        for component in components:
            endpoints = [node for node in component if len(adjacency[node]) == 1]
            # A closed loop has no endpoint. Its nodes are fallback anchors.
            component_endpoints.append(endpoints or component[::max(1, len(component) // 32)])

        best: tuple[float, int, int, float] | None = None
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                for node_a in component_endpoints[i]:
                    ra, ca = coordinates[node_a]
                    tan_a = endpoint_tangent(node_a, coordinates, adjacency)
                    for node_b in component_endpoints[j]:
                        rb, cb = coordinates[node_b]
                        dr, dc = rb - ra, cb - ca
                        gap_px = math.hypot(dr, dc)
                        gap_mm = gap_px / pixels_per_mm
                        if gap_mm > max_gap_mm:
                            continue
                        tan_b = endpoint_tangent(node_b, coordinates, adjacency)
                        if gap_px:
                            ur, uc = dr / gap_px, dc / gap_px
                            align_a = max(-1.0, min(1.0, tan_a[0] * ur + tan_a[1] * uc))
                            align_b = max(-1.0, min(1.0, tan_b[0] * -ur + tan_b[1] * -uc))
                        else:
                            align_a = align_b = 1.0
                        score = gap_px + 2.0 * ((1.0 - align_a) + (1.0 - align_b))
                        if best is None or score < best[0]:
                            best = (score, node_a, node_b, gap_mm)

        if best is None:
            raise ValueError(
                f"The projected skeleton has {len(components)} disconnected parts and no endpoint pair "
                f"within MAX_BRIDGE_GAP_MM={max_gap_mm:.3f}. Increase that setting or inspect the STL."
            )

        _, node_a, node_b, gap_mm = best
        coordinates, adjacency = build_graph(image)
        a, b = coordinates[node_a], coordinates[node_b]
        bridge_pixels = bresenham(a, b)
        width_a = max(min_width_mm, min(max_width_mm, 2.0 * float(distance[a]) / pixels_per_mm))
        width_b = max(min_width_mm, min(max_width_mm, 2.0 * float(distance[b]) / pixels_per_mm))
        bridge_width = min(max_width_mm, max(min_width_mm, gap_mm + (width_a + width_b) * 0.5))
        for row, col in bridge_pixels:
            image[row, col] = True
            bridge_widths[(row, col)] = bridge_width
        bridges += 1


def route_eulerian(
    coordinates: Sequence[tuple[int, int]],
    adjacency: Sequence[Sequence[int]],
    junction_pairings: dict[int, list[tuple[int, int]]],
    bottom_start: bool,
    distance: np.ndarray,
    pixels_per_mm: float,
    max_gap_mm: float,
    min_width_mm: float,
    max_width_mm: float,
) -> tuple[list[int], list[int], list[int], list[float | None], int]:
    """Make an Euler path, adding short straight bridges at odd-degree nodes."""
    base_edges: list[tuple[int, int]] = []
    for node, neighbors in enumerate(adjacency):
        for neighbor in neighbors:
            if node < neighbor:
                base_edges.append((node, neighbor))

    odd = [node for node, neighbors in enumerate(adjacency) if len(neighbors) % 2]
    virtual_edges: list[tuple[int, int]] = []
    virtual_widths: list[float] = []
    if len(odd) > 2:
        # Pair nearby odd-degree endpoints/junctions with short connectors.
        # Any two open ends farther apart than the configured gap remain the
        # start and end of the final path.
        unpaired = set(odd)
        max_gap_px = max_gap_mm * pixels_per_mm
        candidates: list[tuple[float, int, int, float]] = []
        for i, first in enumerate(sorted(unpaired)):
            r0, c0 = coordinates[first]
            for second in sorted(unpaired)[i + 1:]:
                r1, c1 = coordinates[second]
                dr, dc = r1 - r0, c1 - c0
                gap_px = math.hypot(dr, dc)
                if gap_px > max_gap_px:
                    continue
                if gap_px:
                    direction = (dr / gap_px, dc / gap_px)
                    def best_alignment(node: int, target: tuple[float, float]) -> float:
                        neighbor_directions = []
                        nr, nc = coordinates[node]
                        for neighbor in adjacency[node]:
                            rr, cc = coordinates[neighbor]
                            length = math.hypot(rr - nr, cc - nc) or 1.0
                            neighbor_directions.append(((rr - nr) / length, (cc - nc) / length))
                        if len(neighbor_directions) == 1:
                            # At a terminal, continue outwards away from its only edge.
                            return max(-1.0, min(1.0, -sum(a * b for a, b in zip(neighbor_directions[0], target))))
                        return max(
                            max(-1.0, min(1.0, sum(a * b for a, b in zip(candidate, target))))
                            for candidate in neighbor_directions
                        )
                    align_a = best_alignment(first, direction)
                    align_b = best_alignment(second, (-direction[0], -direction[1]))
                else:
                    align_a = align_b = 1.0
                gap_mm = gap_px / pixels_per_mm
                score = gap_px + 2.0 * ((1.0 - align_a) + (1.0 - align_b))
                candidates.append((score, first, second, gap_mm))

        candidates.sort()
        paired: set[int] = set()
        for _, first, second, gap_mm in candidates:
            if first in paired or second in paired:
                continue
            virtual_edges.append((first, second))
            r0, c0 = coordinates[first]
            r1, c1 = coordinates[second]
            width_a = max(min_width_mm, min(max_width_mm, 2.0 * float(distance[r0, c0]) / pixels_per_mm))
            width_b = max(min_width_mm, min(max_width_mm, 2.0 * float(distance[r1, c1]) / pixels_per_mm))
            bridge_width = min(max_width_mm, max(min_width_mm, gap_mm + (width_a + width_b) * 0.5))
            virtual_widths.append(bridge_width)
            paired.update((first, second))

        unpaired.difference_update(paired)
        if len(unpaired) not in (0, 2):
            missing = len(unpaired)
            raise ValueError(
                f"{missing} odd skeleton junction(s) remain after gap pairing; set MAX_BRIDGE_GAP_MM="
                f"{max_gap_mm:.3f}. Increase that setting or inspect the STL."
            )

    all_edges = list(base_edges) + virtual_edges
    all_edge_bridge_widths: list[float | None] = [None] * len(base_edges) + virtual_widths

    edge_adjacency: list[list[int]] = [[] for _ in coordinates]
    for edge_id, (first, second) in enumerate(all_edges):
        edge_adjacency[first].append(edge_id)
        edge_adjacency[second].append(edge_id)

    if not all_edges:
        raise ValueError("The skeleton graph has no printable edges")
    # Turn each user's local arm choice into a fixed pairing of incident edges.
    # Remaining arms are paired by their straightest continuation. This makes
    # the selected crossing connections part of the route itself rather than a
    # loose preference during edge traversal.
    base_edge_by_pair = {
        (min(first, second), max(first, second)): edge_id
        for edge_id, (first, second) in enumerate(base_edges)
    }
    transitions: dict[tuple[int, int], int] = {}
    open_half_edges: list[tuple[int, int]] = []
    for node, incident_edges in enumerate(edge_adjacency):
        remaining = set(incident_edges)
        for first_arm, second_arm in junction_pairings.get(node, []):
            first_edge = base_edge_by_pair.get((min(node, first_arm), max(node, first_arm)))
            second_edge = base_edge_by_pair.get((min(node, second_arm), max(node, second_arm)))
            if first_edge is None or second_edge is None or first_edge == second_edge:
                raise ValueError(f"Invalid crossing-arm selection at skeleton node {node}")
            if first_edge not in remaining or second_edge not in remaining:
                raise ValueError(f"A line end was paired more than once at skeleton node {node}")
            transitions[(node, first_edge)] = second_edge
            transitions[(node, second_edge)] = first_edge
            remaining.remove(first_edge)
            remaining.remove(second_edge)

        def outward_vector(edge_id: int) -> tuple[float, float]:
            first, second = all_edges[edge_id]
            neighbor = second if first == node else first
            row, col = coordinates[node]
            next_row, next_col = coordinates[neighbor]
            length = math.hypot(next_row - row, next_col - col) or 1.0
            return (next_row - row) / length, (next_col - col) / length

        while len(remaining) >= 2:
            best_pair: tuple[float, int, int] | None = None
            for first_edge in remaining:
                first_vector = outward_vector(first_edge)
                for second_edge in remaining:
                    if second_edge <= first_edge:
                        continue
                    second_vector = outward_vector(second_edge)
                    # Opposing outward directions make a smooth continuation.
                    score = first_vector[0] * second_vector[0] + first_vector[1] * second_vector[1]
                    if best_pair is None or score < best_pair[0]:
                        best_pair = (score, first_edge, second_edge)
            assert best_pair is not None
            _, first_edge, second_edge = best_pair
            transitions[(node, first_edge)] = second_edge
            transitions[(node, second_edge)] = first_edge
            remaining.remove(first_edge)
            remaining.remove(second_edge)
        if remaining:
            open_half_edges.append((node, remaining.pop()))

    if len(open_half_edges) not in (0, 2):
        raise ValueError(
            f"The selected crossing pairings leave {len(open_half_edges)} open line ends. "
            "Pair more arms in the review windows so the result forms one route."
        )

    if open_half_edges:
        start_node, first_edge = max(
            open_half_edges,
            key=lambda half_edge: coordinates[half_edge[0]][0] if bottom_start else -coordinates[half_edge[0]][0],
        )
    else:
        start_node = max(range(len(coordinates)), key=lambda n: coordinates[n][0]) if bottom_start else 0
        first_edge = min(edge_adjacency[start_node])

    visited: set[int] = set()
    nodes = [start_node]
    edge_ids: list[int] = []
    current_node, current_edge = start_node, first_edge
    while current_edge not in visited:
        visited.add(current_edge)
        first, second = all_edges[current_edge]
        next_node = second if first == current_node else first
        edge_ids.append(current_edge)
        nodes.append(next_node)
        next_edge = transitions.get((next_node, current_edge))
        if next_edge is None or next_edge in visited:
            break
        current_node, current_edge = next_node, next_edge

    if len(visited) != len(all_edges):
        edge_transitions: list[set[int]] = [set() for _ in all_edges]
        for (_node, first_edge), second_edge in transitions.items():
            edge_transitions[first_edge].add(second_edge)
            edge_transitions[second_edge].add(first_edge)
        trail_edges: list[list[tuple[tuple[int, int], tuple[int, int]]]] = []
        trail_components: list[set[int]] = []
        unassigned = set(range(len(all_edges)))
        while unassigned:
            seed = unassigned.pop()
            component = {seed}
            stack = [seed]
            while stack:
                edge_id = stack.pop()
                for neighbor_edge in edge_transitions[edge_id]:
                    if neighbor_edge in unassigned:
                        unassigned.remove(neighbor_edge)
                        component.add(neighbor_edge)
                        stack.append(neighbor_edge)
            trail_components.append(component)
            trail_edges.append([
                (coordinates[all_edges[edge_id][0]], coordinates[all_edges[edge_id][1]])
                for edge_id in sorted(component)
            ])
        trail_endpoints: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for trail_index, component_edges in enumerate(trail_components):
            open_points = [
                coordinates[node]
                for edge_id in component_edges
                for node in all_edges[edge_id]
                if (node, edge_id) not in transitions
            ]
            if len(open_points) >= 2:
                trail_endpoints.append((open_points[0], open_points[1]))
            elif open_points:
                trail_endpoints.append((open_points[0], open_points[0]))
            else:
                trail_endpoints.append((trail_edges[trail_index][0][0], trail_edges[trail_index][0][0]))
        crossings = [
            (crossing_index, coordinates[node])
            for crossing_index, node in enumerate(
                (node for node, neighbors in enumerate(adjacency) if len(neighbors) >= 3),
                start=1,
            )
        ]
        raise DisconnectedTrailsError(trail_edges, trail_endpoints, crossings)
    segment_multiplicity = [1] * len(edge_ids)
    segment_widths = [all_edge_bridge_widths[edge_id] for edge_id in edge_ids]
    return nodes, edge_ids, segment_multiplicity, segment_widths, len(virtual_edges)


class DisconnectedTrailsError(ValueError):
    """Route failure that includes the edge groups for its separate trails."""

    def __init__(
        self,
        trail_edges: list[list[tuple[tuple[int, int], tuple[int, int]]]],
        trail_endpoints: list[tuple[tuple[int, int], tuple[int, int]]],
        crossings: list[tuple[int, tuple[int, int]]],
    ) -> None:
        self.trail_edges = trail_edges
        self.trail_endpoints = trail_endpoints
        self.crossings = crossings
        super().__init__(
            f"The selected crossing connections create {len(trail_edges)} separate trails. "
            "Adjust the pairings in the crossing review until they form one continuous trail."
        )


def render_disconnected_trails(
    trails: Sequence[Sequence[tuple[tuple[int, int], tuple[int, int]]]],
    endpoints: Sequence[tuple[tuple[int, int], tuple[int, int]]],
    crossings: Sequence[tuple[int, tuple[int, int]]],
) -> str:
    """Render a compact top-down overview, coloring each distinct trail."""
    palette = ("#ff6b6b", "#4dabf7", "#69db7c", "#ffd43b", "#b197fc", "#38d9a9", "#f783ac", "#ffa94d")
    points = [point for trail in trails for edge in trail for point in edge]
    if not points:
        return ""
    min_row = min(point[0] for point in points)
    max_row = max(point[0] for point in points)
    min_col = min(point[1] for point in points)
    max_col = max(point[1] for point in points)
    width = max(1, max_col - min_col)
    height = max(1, max_row - min_row)
    lines: list[str] = []
    legend: list[str] = []
    for index, trail in enumerate(trails):
        color = palette[index % len(palette)]
        segments = "".join(
            f'<line x1="{first[1]}" y1="{first[0]}" x2="{second[1]}" y2="{second[0]}"/>'
            for first, second in trail
        )
        lines.append(f'<g stroke="{color}" stroke-width="1.8">{segments}</g>')
        legend.append(
            f'<span><i style="background:{color}"></i>Trail {index + 1}</span>'
        )
    marker_radius = max(3.0, min(width, height) * 0.012)
    markers = "".join(
        f'<g class="crossing-marker" data-crossing-index="{number - 1}" role="button" '
        f'tabindex="0" aria-label="Edit crossing {number}">'
        f'<circle cx="{col}" cy="{row}" r="{marker_radius:.2f}" fill="#fff" '
        'stroke="#111827" stroke-width="1.5"/>'
        f'<text x="{col}" y="{row}" text-anchor="middle" dominant-baseline="central" '
        f'font-family="sans-serif" font-size="{marker_radius * 1.15:.2f}" font-weight="700" '
        f'fill="#111827">{number}</text></g>'
        for number, (row, col) in crossings
    )
    start_end_markers = "".join(
        f'<g><circle cx="{start[1]}" cy="{start[0]}" r="{marker_radius * 1.25:.2f}" '
        'fill="#69db7c" stroke="#111827" stroke-width="1.5"/>'
        f'<text x="{start[1]}" y="{start[0]}" text-anchor="middle" dominant-baseline="central" '
        f'font-family="sans-serif" font-size="{marker_radius * 1.15:.2f}" font-weight="700" '
        'fill="#102018">S</text>'
        f'<rect x="{end[1] - marker_radius:.2f}" y="{end[0] - marker_radius:.2f}" '
        f'width="{marker_radius * 2:.2f}" height="{marker_radius * 2:.2f}" rx="1" '
        'fill="#ff6b6b" stroke="#111827" stroke-width="1.5"/>'
        f'<text x="{end[1]}" y="{end[0]}" text-anchor="middle" dominant-baseline="central" '
        f'font-family="sans-serif" font-size="{marker_radius * 1.15:.2f}" font-weight="700" '
        'fill="#3b0a0a">E</text></g>'
        for start, end in endpoints
    )
    return (
        '<section><h2>Path overview</h2>'
        '<p>Each color is one trail. S/E mark its start and end. Click a numbered crossing marker to edit that crossing.</p>'
        f'<svg viewBox="{min_col} {min_row} {width} {height}" role="img" '
        'aria-label="Top-down overview of separated path trails">'
        f'{"".join(lines)}{markers}{start_end_markers}</svg><div class="legend">{"".join(legend)}'
        '<span><i style="background:#69db7c"></i>Start</span>'
        '<span><i style="background:#ff6b6b"></i>End</span></div></section>'
    )


# ------------------------------- G-code output -------------------------------

def bead_area(width_mm: float, height_mm: float) -> float:
    """Rounded-rectangle bead area used by common slicer flow calculations."""
    if width_mm >= height_mm:
        return height_mm * (width_mm - height_mm) + math.pi * height_mm * height_mm / 4.0
    return math.pi * width_mm * width_mm / 4.0


def interpolate_path(
    points: Sequence[tuple[float, float]],
    widths: Sequence[float],
    segment_multiplicity: Sequence[int],
    spacing_mm: float,
) -> tuple[list[tuple[float, float]], list[float], list[int]]:
    out_points = [points[0]]
    out_widths = [widths[0]]
    out_multiplicity: list[int] = []
    for index in range(len(points) - 1):
        x0, y0 = points[index]
        x1, y1 = points[index + 1]
        w0, w1 = widths[index], widths[index + 1]
        length = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, int(math.ceil(length / spacing_mm)))
        for step in range(1, steps + 1):
            t = step / steps
            out_points.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
            out_widths.append(w0 + (w1 - w0) * t)
            out_multiplicity.append(max(1, segment_multiplicity[index]))
    return out_points, out_widths, out_multiplicity


def expand_and_simplify_route(
    points: Sequence[tuple[float, float]],
    widths: Sequence[float],
    segment_multiplicity: Sequence[int],
    segment_width_overrides: Sequence[float | None],
    position_tolerance: float,
    width_tolerance: float,
) -> tuple[list[tuple[float, float]], list[float], list[int]]:
    """Preserve bridge-width changes, then remove redundant raster points."""
    expanded_points = [points[0]]
    expanded_widths = [widths[0]]
    expanded_multiplicity: list[int] = []
    for index in range(len(points) - 1):
        first, second = points[index], points[index + 1]
        override = segment_width_overrides[index]
        if override is not None:
            midpoint = ((first[0] + second[0]) * 0.5, (first[1] + second[1]) * 0.5)
            expanded_points.append(midpoint)
            expanded_widths.append(float(override))
            expanded_multiplicity.append(segment_multiplicity[index])
        expanded_points.append(second)
        expanded_widths.append(widths[index + 1])
        expanded_multiplicity.append(segment_multiplicity[index])

    count = len(expanded_points)
    if count <= 2:
        return expanded_points, expanded_widths, expanded_multiplicity
    cumulative = [0.0]
    for first, second in zip(expanded_points, expanded_points[1:]):
        cumulative.append(cumulative[-1] + math.dist(first, second))

    keep = {0, count - 1}
    # Keep both sides of any extrusion-rate change.
    for index in range(1, count - 1):
        if expanded_multiplicity[index - 1] != expanded_multiplicity[index]:
            keep.add(index)
    anchors = sorted(keep)
    stack = [(anchors[i], anchors[i + 1]) for i in range(len(anchors) - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        x0, y0 = expanded_points[start]
        x1, y1 = expanded_points[end]
        vx, vy = x1 - x0, y1 - y0
        length_sq = vx * vx + vy * vy
        arc_length = cumulative[end] - cumulative[start]
        best_error = 1.0
        best_index = -1
        for index in range(start + 1, end):
            x, y = expanded_points[index]
            if length_sq:
                t_geom = max(0.0, min(1.0, ((x - x0) * vx + (y - y0) * vy) / length_sq))
                projected_x, projected_y = x0 + t_geom * vx, y0 + t_geom * vy
            else:
                projected_x, projected_y = x0, y0
            position_error = math.hypot(x - projected_x, y - projected_y) / position_tolerance
            t_width = (cumulative[index] - cumulative[start]) / arc_length if arc_length else 0.0
            expected_width = expanded_widths[start] + (expanded_widths[end] - expanded_widths[start]) * t_width
            width_error = abs(expanded_widths[index] - expected_width) / width_tolerance
            error = max(position_error, width_error)
            if error > best_error:
                best_error = error
                best_index = index
        if best_index >= 0:
            keep.add(best_index)
            stack.extend(((start, best_index), (best_index, end)))

    selected = sorted(keep)
    reduced_points = [expanded_points[index] for index in selected]
    reduced_widths = [expanded_widths[index] for index in selected]
    reduced_multiplicity = [
        max(expanded_multiplicity[start:end])
        for start, end in zip(selected, selected[1:])
    ]
    return reduced_points, reduced_widths, reduced_multiplicity


def build_gcode(
    name: str,
    points: Sequence[tuple[float, float]],
    widths: Sequence[float],
    segment_multiplicity: Sequence[int],
    bridge_count: int,
    direction: str,
    layer_height_mm: float,
    flow_multiplier_percent: float,
    filament_diameter_mm: float,
) -> str:
    if direction == "reverse":
        points = list(reversed(points))
        widths = list(reversed(widths))
        segment_multiplicity = list(reversed(segment_multiplicity))

    filament_area = math.pi * (filament_diameter_mm * 0.5) ** 2
    travel_feed = int(round(TRAVEL_SPEED_MM_S * 60.0))
    lines = [
        "; BEGIN VARIABLE_WIDTH_LINEART",
        f"; {name} — {direction} direction",
        f"; Layer height {layer_height_mm:.4f} mm; host flow multiplier {flow_multiplier_percent:.2f}%.",
        f"; Speed target {PRINT_SPEED_MM_S:.2f} mm/s; max flow {MAX_FLOW_RATE_MM3_S:.3f} mm^3/s; bridges: {bridge_count}.",
        f"G0 X{points[0][0]:.3f} Y{points[0][1]:.3f} F{travel_feed}",
    ]

    for index in range(len(points) - 1):
        x0, y0 = points[index]
        x1, y1 = points[index + 1]
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 1e-6:
            continue
        width = max(MIN_EXTRUSION_WIDTH_MM, min(MAX_EXTRUSION_WIDTH_MM, (widths[index] + widths[index + 1]) * 0.5))
        area = bead_area(width, layer_height_mm)
        passes = max(1, segment_multiplicity[index])
        if passes > 1:
            shared_area = area / passes
            minimum_area = bead_area(MIN_EXTRUSION_WIDTH_MM, layer_height_mm)
            area = max(shared_area, minimum_area)
        speed = min(PRINT_SPEED_MM_S, MAX_FLOW_RATE_MM3_S * FLOW_SAFETY_FACTOR / area)
        # M221 scales commanded E at the printer. Divide by its multiplier to
        # preserve the intended bead volume when the host has e.g. M221 S95.
        e_delta = area * length / (filament_area * (flow_multiplier_percent / 100.0))
        lines.append(
            f"G1 X{x1:.3f} Y{y1:.3f} E{e_delta:.5f} F{int(math.floor(speed * 60.0))}"
        )
    lines.append("; END VARIABLE_WIDTH_LINEART")
    lines.append("")
    return "\n".join(lines)


def normalize_setting_name(name: str) -> str:
    return "_".join(name.strip().lower().replace("-", " ").replace("_", " ").split())


def parse_prusaslicer_settings(gcode_text: str) -> dict[str, str]:
    """Read PrusaSlicer's trailing ASCII config comments."""
    settings: dict[str, str] = {}
    for line in gcode_text.splitlines():
        match = re.match(r"^\s*;\s*([^;=]+?)\s*=\s*(.*?)\s*$", line)
        if match:
            settings[normalize_setting_name(match.group(1))] = match.group(2)
    return settings


def first_setting_number(settings: dict[str, str], *names: str) -> float | None:
    """Read the first number from a scalar or comma-separated slicer setting."""
    for name in names:
        value = settings.get(normalize_setting_name(name))
        if value is None:
            continue
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value)
        if match:
            return float(match.group(0))
    return None


def first_positive_setting_number(settings: dict[str, str], *names: str) -> float | None:
    for name in names:
        value = first_setting_number(settings, name)
        if value is not None and value > 0:
            return value
    return None


def resolve_profile_parameters(settings: dict[str, str]) -> None:
    """Fill automatic extrusion settings from PrusaSlicer's exported config."""
    global MIN_EXTRUSION_WIDTH_MM
    global MAX_FLOW_RATE_MM3_S
    global FLOW_SAFETY_FACTOR
    global PRINT_SPEED_MM_S
    global TRAVEL_SPEED_MM_S

    if MIN_EXTRUSION_WIDTH_MM is None:
        MIN_EXTRUSION_WIDTH_MM = first_positive_setting_number(
            settings,
            "external_perimeter_extrusion_width",
            "external_perimeters_extrusion_width",
            "extrusion_width",
        )
        if MIN_EXTRUSION_WIDTH_MM is None or MIN_EXTRUSION_WIDTH_MM <= 0:
            raise ValueError(
                "Could not find a positive external perimeter width in the PrusaSlicer settings. "
                "Set MIN_EXTRUSION_WIDTH_MM explicitly at the top of the script."
            )

    if MAX_FLOW_RATE_MM3_S is None:
        flow_limits = [
            value
            for value in (
                first_setting_number(settings, "filament_max_volumetric_speed"),
                first_setting_number(settings, "max_volumetric_speed"),
            )
            if value is not None and value > 0
        ]
        if not flow_limits:
            raise ValueError(
                "Could not find a positive filament_max_volumetric_speed or max_volumetric_speed "
                "in the PrusaSlicer settings. Set MAX_FLOW_RATE_MM3_S explicitly."
            )
        MAX_FLOW_RATE_MM3_S = min(flow_limits)

    if PRINT_SPEED_MM_S is None:
        external_speed = settings.get("external_perimeter_speed", "")
        if external_speed:
            speed_value = first_setting_number(settings, "external_perimeter_speed")
            if speed_value is not None and speed_value > 0:
                if "%" in external_speed:
                    baseline = first_setting_number(settings, "perimeter_speed", "max_print_speed")
                    if baseline is not None and baseline > 0:
                        PRINT_SPEED_MM_S = baseline * speed_value / 100.0
                else:
                    PRINT_SPEED_MM_S = speed_value
        if PRINT_SPEED_MM_S is None:
            PRINT_SPEED_MM_S = first_setting_number(settings, "perimeter_speed", "max_print_speed")
        if PRINT_SPEED_MM_S is None or PRINT_SPEED_MM_S <= 0:
            raise ValueError(
                "Could not find external_perimeter_speed or perimeter_speed in the PrusaSlicer settings. "
                "Set PRINT_SPEED_MM_S explicitly."
            )

    if TRAVEL_SPEED_MM_S is None:
        TRAVEL_SPEED_MM_S = first_setting_number(settings, "travel_speed")
        if TRAVEL_SPEED_MM_S is None or TRAVEL_SPEED_MM_S <= 0:
            raise ValueError(
                "Could not find travel_speed in the PrusaSlicer settings. "
                "Set TRAVEL_SPEED_MM_S explicitly."
            )

    if FLOW_SAFETY_FACTOR is None:
        custom_margin = first_setting_number(settings, "flow_safety_factor")
        FLOW_SAFETY_FACTOR = (
            custom_margin if custom_margin is not None else FLOW_SAFETY_FACTOR_FALLBACK
        )


def find_filament_diameter(gcode_text: str) -> float:
    if FILAMENT_DIAMETER_MM_OVERRIDE is not None:
        return FILAMENT_DIAMETER_MM_OVERRIDE

    for line in gcode_text.splitlines():
        match = re.match(r"\s*;\s*filament_diameter\s*=\s*([\d.]+)", line, re.IGNORECASE)
        if match and float(match.group(1)) > 0:
            return float(match.group(1))
    return FILAMENT_DIAMETER_FALLBACK_MM


def remove_previous_lineart(lines: Sequence[str]) -> list[str]:
    """Make reruns replace this tool's marked blocks instead of stacking them."""
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "; BEGIN VARIABLE_WIDTH_LINEART":
            end = i + 1
            while end < len(lines) and lines[end].strip() != "; END VARIABLE_WIDTH_LINEART":
                end += 1
            if end == len(lines):
                raise ValueError("Found an unterminated generated line-art block in the input G-code")
            i = end + 1
            continue

        legacy_header = re.match(r"^;\s+.+\s+—\s+(?:forward|reverse)\s+direction\s*$", line)
        legacy_comment = (
            i + 1 < len(lines)
            and "Z is omitted: insert at the desired layer height." in lines[i + 1]
        )
        if legacy_header and legacy_comment:
            # Previous script versions emitted an unmarked one-layer snippet.
            # Its generated moves occupied the remainder of that layer.
            i += 2
            while i < len(lines) and not re.match(r"^\s*;\s*LAYER_CHANGE\b", lines[i], re.IGNORECASE):
                i += 1
            continue

        result.append(lines[i])
        i += 1
    return result


def layer_sections(lines: Sequence[str]) -> list[tuple[int, int, float | None]]:
    starts = [
        i for i, line in enumerate(lines)
        if re.match(r"^\s*;\s*LAYER_CHANGE\b", line, re.IGNORECASE)
    ]
    sections: list[tuple[int, int, float | None]] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(lines)
        height = None
        for line in lines[start:end]:
            match = re.match(r"^\s*;\s*HEIGHT\s*:\s*([\d.]+)", line, re.IGNORECASE)
            if match:
                height = float(match.group(1))
                break
        sections.append((start, end, height))
    return sections


def host_extrusion_state(lines: Sequence[str], stop: int) -> float:
    """Return M221 flow and ensure supported coordinate/extrusion modes."""
    absolute_xy: bool | None = None
    relative_e: bool | None = None
    millimeters: bool | None = None
    flow_percent = 100.0

    for line in lines[:stop]:
        command = line.split(";", 1)[0].strip().upper()
        if not command:
            continue
        words = command.split()
        opcode = words[0]
        if opcode == "G90":
            absolute_xy = True
        elif opcode == "G91":
            absolute_xy = False
        elif opcode == "G21":
            millimeters = True
        elif opcode == "G20":
            millimeters = False
        elif opcode == "M83":
            relative_e = True
        elif opcode == "M82":
            relative_e = False
        elif opcode == "M221":
            match = re.search(r"(?:^|\s)S([-+]?[\d.]+)", command)
            if match:
                flow_percent = float(match.group(1))

    if absolute_xy is not True or relative_e is not True or millimeters is not True:
        raise ValueError(
            "The host G-code must use millimeter units, absolute XY (G90), and relative extrusion (M83) "
            "at the layer insertion point."
        )
    if flow_percent <= 0:
        raise ValueError(f"Invalid active M221 flow multiplier: {flow_percent:g}%")
    return flow_percent


def beginning_of_printing_moves(lines: Sequence[str], start: int, end: int) -> int:
    """Locate the first XY extrusion, after the layer's Z setup and priming."""
    for index in range(start, end):
        command = lines[index].split(";", 1)[0].strip().upper()
        if not command.startswith(("G0 ", "G1 ")):
            continue
        has_xy = re.search(r"(?:^|\s)[XY][-+]?[\d.]+(?:\s|$)", command)
        extrusion = re.search(r"(?:^|\s)E([-+]?[\d.]+)(?:\s|$)", command)
        if has_xy and extrusion and float(extrusion.group(1)) > 0:
            return index
    # Older generated paths sometimes occupied the whole first layer. Once
    # removed, the section ends after its remaining setup/prime moves, which is
    # the correct insertion position for a now-empty layer.
    return end


def xy_position_at(lines: Sequence[str], end: int) -> tuple[float, float] | None:
    """Track the commanded XY position before an insertion point."""
    x: float | None = None
    y: float | None = None
    absolute = True
    for line in lines[:end]:
        command = line.split(";", 1)[0].strip().upper()
        if command == "G90":
            absolute = True
            continue
        if command == "G91":
            absolute = False
            continue
        if not command.startswith(("G0 ", "G1 ", "G92 ")):
            continue
        values = {
            axis: float(match.group(1))
            for axis in "XY"
            if (match := re.search(rf"(?:^|\s){axis}([-+]?[\d.]+)(?:\s|$)", command))
        }
        if command.startswith("G92 "):
            if "X" in values:
                x = values["X"]
            if "Y" in values:
                y = values["Y"]
        elif absolute:
            x = values.get("X", x)
            y = values.get("Y", y)
        else:
            x = x + values["X"] if x is not None and "X" in values else x
            y = y + values["Y"] if y is not None and "Y" in values else y
    return (x, y) if x is not None and y is not None else None


def inject_alternating_paths(
    source_lines: Sequence[str],
    name: str,
    points: Sequence[tuple[float, float]],
    widths: Sequence[float],
    segment_multiplicity: Sequence[int],
    bridge_count: int,
    filament_diameter_mm: float,
) -> tuple[list[str], list[tuple[int, float]]]:
    lines = remove_previous_lineart(source_lines)
    sections = layer_sections(lines)
    if not sections:
        raise ValueError("No ;LAYER_CHANGE markers were found in the host G-code")
    if START_LAYER < 1 or LAYER_COUNT < 1:
        raise ValueError("START_LAYER and LAYER_COUNT must both be positive")
    if LAYER_COUNT % 2:
        raise ValueError("LAYER_COUNT must be even so every injection contains a forward/reverse pair")
    first = START_LAYER - 1
    last = first + LAYER_COUNT
    if last > len(sections):
        raise ValueError(
            f"Requested layers {START_LAYER} through {last}, but the host has only {len(sections)} layer markers"
        )

    output = list(lines[:sections[0][0]])
    inserted: list[tuple[int, float]] = []
    injection_start_xy: tuple[float, float] | None = None
    for layer_index, (start, end, height) in enumerate(sections):
        if not first <= layer_index < last:
            output.extend(lines[start:end])
            continue
        if height is None:
            raise ValueError(
                f"Layer {layer_index + 1} has no ;HEIGHT value; cannot calculate width-specific feedrates"
            )
        if height <= 0:
            raise ValueError(f"Layer {layer_index + 1} has invalid height {height:g} mm")
        direction = "forward" if (layer_index - first) % 2 == 0 else "reverse"
        insertion_point = "end" if direction == "forward" else "beginning"
        insertion_index = (
            beginning_of_printing_moves(lines, start, end)
            if insertion_point == "beginning"
            else end
        )
        flow_percent = host_extrusion_state(lines, insertion_index)
        host_resume_xy = xy_position_at(lines, insertion_index)
        if direction == "forward":
            injection_start_xy = host_resume_xy
        snippet = build_gcode(
            name,
            points,
            widths,
            segment_multiplicity,
            bridge_count,
            direction,
            height,
            flow_percent,
            filament_diameter_mm,
        ).splitlines()
        if direction == "reverse":
            if injection_start_xy is None:
                raise ValueError("Could not determine the XY position where this path pair began")
            travel_feed = int(round(TRAVEL_SPEED_MM_S * 60.0))
            if snippet and not snippet[-1]:
                snippet.pop()
            end_marker = snippet.pop()
            snippet.extend((
                "; Return to the XY position where this path pair began.",
                f"G0 X{injection_start_xy[0]:.3f} Y{injection_start_xy[1]:.3f} F{travel_feed}",
            ))
            if host_resume_xy is not None and any(
                abs(a - b) > 0.001 for a, b in zip(injection_start_xy, host_resume_xy)
            ):
                snippet.extend((
                    "; Restore the host layer's original XY start position.",
                    f"G0 X{host_resume_xy[0]:.3f} Y{host_resume_xy[1]:.3f} F{travel_feed}",
                ))
            snippet.extend((end_marker, ""))
        output.extend(lines[start:insertion_index])
        output.extend(snippet)
        output.extend(lines[insertion_index:end])
        inserted.append((layer_index + 1, height))
    return output, inserted


def main() -> None:
    if not INPUT_STL.is_file():
        raise FileNotFoundError(f"Input STL not found: {INPUT_STL.resolve()}")
    if not INPUT_GCODE_FILE.is_file():
        raise FileNotFoundError(f"Input G-code not found: {INPUT_GCODE_FILE.resolve()}")
    if INPUT_GCODE_FILE.resolve() == OUTPUT_GCODE_FILE.resolve():
        raise ValueError("OUTPUT_GCODE_FILE must differ from INPUT_GCODE_FILE so the source is preserved")
    if INPUT_GCODE_FILE.suffix.lower() != ".gcode" or OUTPUT_GCODE_FILE.suffix.lower() != ".gcode":
        raise ValueError("INPUT_GCODE_FILE and OUTPUT_GCODE_FILE must use the .gcode extension")
    if START_LAYER < 1 or LAYER_COUNT < 1:
        raise ValueError("START_LAYER and LAYER_COUNT must both be positive")
    if MAX_EXTRUSION_WIDTH_MM <= 0:
        raise ValueError("MAX_EXTRUSION_WIDTH_MM must be positive")
    if MIN_EXTRUSION_WIDTH_MM is not None and MIN_EXTRUSION_WIDTH_MM <= 0:
        raise ValueError("MIN_EXTRUSION_WIDTH_MM must be positive when explicitly set")
    if MAX_FLOW_RATE_MM3_S is not None and MAX_FLOW_RATE_MM3_S <= 0:
        raise ValueError("MAX_FLOW_RATE_MM3_S must be positive when explicitly set")
    if PRINT_SPEED_MM_S is not None and PRINT_SPEED_MM_S <= 0:
        raise ValueError("PRINT_SPEED_MM_S must be positive when explicitly set")
    if TRAVEL_SPEED_MM_S is not None and TRAVEL_SPEED_MM_S <= 0:
        raise ValueError("TRAVEL_SPEED_MM_S must be positive when explicitly set")
    if FLOW_SAFETY_FACTOR is not None and not 0 < FLOW_SAFETY_FACTOR <= 1:
        raise ValueError("FLOW_SAFETY_FACTOR must be in (0, 1]")
    if not 0 < FLOW_SAFETY_FACTOR_FALLBACK <= 1:
        raise ValueError("FLOW_SAFETY_FACTOR_FALLBACK must be in (0, 1]")
    if FILAMENT_DIAMETER_FALLBACK_MM <= 0 or (
        FILAMENT_DIAMETER_MM_OVERRIDE is not None and FILAMENT_DIAMETER_MM_OVERRIDE <= 0
    ):
        raise ValueError("Filament diameter values must be positive")
    if MAX_BRIDGE_GAP_MM <= 0:
        raise ValueError("MAX_BRIDGE_GAP_MM must be positive")
    if MAX_CROSSING_CLUSTER_SPAN_MM <= 0:
        raise ValueError("MAX_CROSSING_CLUSTER_SPAN_MM must be positive")
    if len(PROJECTION_AXES) != 2 or len(set(PROJECTION_AXES)) != 2 or any(axis not in (0, 1, 2) for axis in PROJECTION_AXES):
        raise ValueError("PROJECTION_AXES must contain two distinct STL axes from 0, 1, 2")
    if (
        OUTPUT_SCALE <= 0
        or RASTER_PIXELS_PER_MM <= 0
        or PATH_SAMPLE_SPACING_MM <= 0
        or PATH_SIMPLIFY_TOLERANCE_MM <= 0
        or WIDTH_SIMPLIFY_TOLERANCE_MM <= 0
    ):
        raise ValueError("Scale and raster/sample resolutions must be positive")

    host_text = INPUT_GCODE_FILE.read_text(encoding="utf-8", errors="replace")
    profile_settings = parse_prusaslicer_settings(host_text)
    resolve_profile_parameters(profile_settings)
    if MIN_EXTRUSION_WIDTH_MM is None or MIN_EXTRUSION_WIDTH_MM > MAX_EXTRUSION_WIDTH_MM:
        raise ValueError("Set positive widths with MAX_EXTRUSION_WIDTH_MM >= MIN_EXTRUSION_WIDTH_MM")
    if MAX_FLOW_RATE_MM3_S is None or MAX_FLOW_RATE_MM3_S <= 0:
        raise ValueError("Maximum flow rate must be positive")
    if PRINT_SPEED_MM_S is None or PRINT_SPEED_MM_S <= 0:
        raise ValueError("Print speed must be positive")
    if TRAVEL_SPEED_MM_S is None or TRAVEL_SPEED_MM_S <= 0:
        raise ValueError("Travel speed must be positive")
    if FLOW_SAFETY_FACTOR is None or not 0 < FLOW_SAFETY_FACTOR <= 1:
        raise ValueError("FLOW_SAFETY_FACTOR must be in (0, 1]")
    filament_diameter_mm = find_filament_diameter(host_text)

    all_triangles = load_stl(INPUT_STL)
    components = split_face_components(all_triangles, VERTEX_WELD_TOLERANCE_MM)
    kept_component_ids = [
        component_id
        for component_id, face_ids in enumerate(components)
        if len(face_ids) >= IGNORE_COMPONENTS_BELOW_TRIANGLES
    ]
    if not kept_component_ids:
        largest = max(components, key=len)
        kept_component_ids = [components.index(largest)]
    kept_face_ids = [face for component_id in kept_component_ids for face in components[component_id]]
    triangles = [all_triangles[face_id] for face_id in kept_face_ids]
    ignored_count = len(all_triangles) - len(triangles)

    mask, bounds, padding = rasterize_projected_triangles(
        triangles, PROJECTION_AXES, RASTER_PIXELS_PER_MM
    )
    if not np.any(mask):
        raise ValueError("No projected mesh surface was rasterized; check PROJECTION_AXES")
    distance = chamfer_distance(mask)
    skeleton = skeletonize(mask)
    skeleton = prune_short_spurs(skeleton, SPUR_PRUNE_LENGTH_MM * RASTER_PIXELS_PER_MM)
    skeleton, bridge_widths, bridge_count = connect_nearby_components(
        skeleton,
        distance,
        RASTER_PIXELS_PER_MM,
        MAX_BRIDGE_GAP_MM,
        MIN_EXTRUSION_WIDTH_MM,
        MAX_EXTRUSION_WIDTH_MM,
    )

    coordinates, adjacency = build_graph(skeleton)
    graph_parts = graph_components(adjacency)
    if len(graph_parts) != 1:
        raise ValueError(f"Could not connect all path components; {len(graph_parts)} remain")
    coordinates, adjacency, merged_crossings = collapse_odd_junction_pairs(
        coordinates,
        adjacency,
        MAX_CROSSING_CLUSTER_SPAN_MM * RASTER_PIXELS_PER_MM,
    )
    if merged_crossings:
        print(
            f"Grouped {merged_crossings} pairs of odd skeleton junctions into even-arm crossings.",
            file=sys.stderr,
        )
    junction_nodes = [node for node, neighbors in enumerate(adjacency) if len(neighbors) >= 3]
    validated_route: tuple[
        list[int], list[int], list[int], list[float | None], int
    ] | None = None

    def validate_crossing_decisions(
        decisions: dict[int, list[tuple[int, int]]],
    ) -> tuple[str | None, str]:
        nonlocal validated_route
        try:
            validated_route = route_eulerian(
                coordinates,
                adjacency,
                decisions,
                bottom_start=True,
                distance=distance,
                pixels_per_mm=RASTER_PIXELS_PER_MM,
                max_gap_mm=MAX_BRIDGE_GAP_MM,
                min_width_mm=MIN_EXTRUSION_WIDTH_MM,
                max_width_mm=MAX_EXTRUSION_WIDTH_MM,
            )
        except ValueError as exc:
            validated_route = None
            overview = (
                render_disconnected_trails(exc.trail_edges, exc.trail_endpoints, exc.crossings)
                if isinstance(exc, DisconnectedTrailsError)
                else ""
            )
            return str(exc), overview
        route_nodes = validated_route[0]
        route_edges = [[
            (coordinates[first], coordinates[second])
            for first, second in zip(route_nodes, route_nodes[1:])
        ]]
        route_endpoints = [(coordinates[route_nodes[0]], coordinates[route_nodes[-1]])]
        crossing_markers = [
            (index, coordinates[node])
            for index, node in enumerate(junction_nodes, start=1)
        ]
        return None, render_disconnected_trails(route_edges, route_endpoints, crossing_markers)

    junction_pairings = (
        review_crossing_pairings(
            skeleton,
            coordinates,
            adjacency,
            junction_nodes,
            RASTER_PIXELS_PER_MM,
            validate_crossing_decisions,
        )
        if REVIEW_CROSSINGS_GRAPHICALLY
        else {}
    )
    if validated_route is not None:
        route, _edge_ids, segment_multiplicity, segment_width_overrides, topology_bridges = validated_route
    else:
        route, _edge_ids, segment_multiplicity, segment_width_overrides, topology_bridges = route_eulerian(
            coordinates,
            adjacency,
            junction_pairings,
            bottom_start=True,
            distance=distance,
            pixels_per_mm=RASTER_PIXELS_PER_MM,
            max_gap_mm=MAX_BRIDGE_GAP_MM,
            min_width_mm=MIN_EXTRUSION_WIDTH_MM,
            max_width_mm=MAX_EXTRUSION_WIDTH_MM,
        )

    u_min, u_max, v_min, v_max = bounds
    center_u, center_v = (u_min + u_max) * 0.5, (v_min + v_max) * 0.5
    route_points: list[tuple[float, float]] = []
    route_widths: list[float] = []
    for index, node in enumerate(route):
        row, col = coordinates[node]
        u = u_min + (col + 0.5 - padding) / RASTER_PIXELS_PER_MM
        v = v_max + (padding - (row + 0.5)) / RASTER_PIXELS_PER_MM
        x = (u - center_u) * OUTPUT_SCALE + OUTPUT_CENTER_X_MM
        y = (v - center_v) * OUTPUT_SCALE + OUTPUT_CENTER_Y_MM
        route_points.append((x, y))
        override = bridge_widths.get((row, col))
        width = override if override is not None else 2.0 * float(distance[row, col]) / RASTER_PIXELS_PER_MM
        width *= OUTPUT_SCALE
        width = max(MIN_EXTRUSION_WIDTH_MM, min(MAX_EXTRUSION_WIDTH_MM, width))
        route_widths.append(width)

    if len(route_points) < 2:
        raise ValueError("The generated path contains fewer than two points")

    route_points, route_widths, segment_multiplicity = expand_and_simplify_route(
        route_points,
        route_widths,
        segment_multiplicity,
        [
            None if width is None else max(
                MIN_EXTRUSION_WIDTH_MM,
                min(MAX_EXTRUSION_WIDTH_MM, width * OUTPUT_SCALE),
            )
            for width in segment_width_overrides
        ],
        PATH_SIMPLIFY_TOLERANCE_MM,
        WIDTH_SIMPLIFY_TOLERANCE_MM,
    )
    # Resample by arclength to cap move length and keep G-code compact.
    points, widths, segment_multiplicity = interpolate_path(
        route_points,
        route_widths,
        segment_multiplicity,
        PATH_SAMPLE_SPACING_MM,
    )
    bridge_count += topology_bridges
    if topology_bridges:
        print(
            f"Connected {topology_bridges} odd-degree skeleton junction pair(s) with short bridge segments.",
            file=sys.stderr,
        )

    base_name = INPUT_STL.stem
    updated_lines, inserted_layers = inject_alternating_paths(
        host_text.splitlines(),
        base_name,
        points,
        widths,
        segment_multiplicity,
        bridge_count,
        filament_diameter_mm,
    )
    updated_text = "\n".join(updated_lines) + "\n"
    OUTPUT_GCODE_FILE.write_text(updated_text, encoding="utf-8")

    print(f"Input facets: {len(all_triangles)}")
    print(f"Ignored facets/components: {ignored_count} facets ignored")
    print(f"Raster: {mask.shape[1]} x {mask.shape[0]} px at {RASTER_PIXELS_PER_MM:g} px/mm")
    print(f"Path points: {len(points)}; bridges added: {bridge_count}")
    print(f"Projected bounds: {u_max-u_min:.3f} x {v_max-v_min:.3f} mm")
    print(
        f"Widths: {MIN_EXTRUSION_WIDTH_MM:.3f}–{MAX_EXTRUSION_WIDTH_MM:.3f} mm; "
        f"flow cap: {MAX_FLOW_RATE_MM3_S:.3f} mm^3/s × {FLOW_SAFETY_FACTOR:.3f}"
    )
    print(f"Print speed: {PRINT_SPEED_MM_S:.2f} mm/s; travel speed: {TRAVEL_SPEED_MM_S:.2f} mm/s")
    print(f"Filament diameter: {filament_diameter_mm:.3f} mm")
    print("Inserted paths into layers: " + ", ".join(
        f"{layer} ({height:.3f} mm, {'forward at layer end' if offset % 2 == 0 else 'reverse at layer beginning'})"
        for offset, (layer, height) in enumerate(inserted_layers)
    ))
    print(f"Wrote {OUTPUT_GCODE_FILE.resolve()}")


if __name__ == "__main__":
    main()
