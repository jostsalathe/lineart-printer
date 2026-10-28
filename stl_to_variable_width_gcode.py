#!/usr/bin/env python3
"""Convert a swept-line STL into two direction-reversed, variable-width G-code paths.

The STL has no centerline metadata. This script projects the largest mesh
components onto a chosen plane, skeletonizes the silhouette, and follows the
resulting medial path. It writes relative-E extrusion moves and deliberately
omits all Z moves so the snippets can be inserted at an existing layer.

Requires NumPy: ``python -m pip install numpy``.
"""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path
from typing import Sequence

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment-dependent message
    raise SystemExit("This script requires NumPy. Install it with: python -m pip install numpy") from exc


# ------------------------------- User inputs --------------------------------

INPUT_FILE = Path("One+Line+Frau.stl")
OUTPUT_FORWARD = Path("One+Line+Frau_forward.gcode")
OUTPUT_REVERSE = Path("One+Line+Frau_reverse.gcode")

MIN_EXTRUSION_WIDTH_MM = 0.65
MAX_EXTRUSION_WIDTH_MM = 1.20
EXTRUSION_HEIGHT_MM = 0.30
MAX_FLOW_RATE_MM3_S = 8.0
FLOW_SAFETY_FACTOR = 0.98  # Allow for rounded E/F values in the emitted G-code.

# Additional printer and path settings.
FILAMENT_DIAMETER_MM = 1.75
PRINT_SPEED_MM_S = 40.0
TRAVEL_SPEED_MM_S = 120.0
# Maximum centerline gap the script may connect. This is separate from the
# bead-width limits: connectors are printed at a capped width. Lower it if you
# want the script to reject uncertain joins rather than bridge them.
MAX_BRIDGE_GAP_MM = 8.0
RASTER_PIXELS_PER_MM = 5.0
PATH_SAMPLE_SPACING_MM = 0.50
PATH_SIMPLIFY_TOLERANCE_MM = 0.20
WIDTH_SIMPLIFY_TOLERANCE_MM = 0.08
SPUR_PRUNE_LENGTH_MM = 0.60
VERTEX_WELD_TOLERANCE_MM = 0.001
IGNORE_COMPONENTS_BELOW_TRIANGLES = 100

# The line art is viewed along STL X, so STL Y/Z become printer X/Y. Change
# these axes if the model uses a different drawing plane. The coordinates are
# centered at the configured target point; adjust this to align with the
# replacement pedestal in the host G-code.
PROJECTION_AXES = (1, 2)  # horizontal, vertical; (0, 1) means view along Z
OUTPUT_CENTER_X_MM = 110.0
OUTPUT_CENTER_Y_MM = 110.0
OUTPUT_SCALE = 1.0

# A conventional 220 x 220 mm bed is assumed for the default output center.
# G-code uses absolute XY (G90) and relative extrusion (M83); the snippet does
# not change either mode because it cannot safely restore unknown host state.


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
    remaining_odd = [node for node, edge_ids in enumerate(edge_adjacency) if len(edge_ids) % 2]
    if remaining_odd:
        bottom_endpoints = sorted(remaining_odd, key=lambda n: coordinates[n][0], reverse=True)
        start = bottom_endpoints[0] if bottom_start else remaining_odd[0]
    else:
        start = max(range(len(coordinates)), key=lambda n: coordinates[n][0]) if bottom_start else 0

    visited = bytearray(len(all_edges))
    stack_nodes = [start]
    stack_edges: list[int] = []
    reverse_nodes: list[int] = []
    reverse_edges: list[int] = []
    while stack_nodes:
        node = stack_nodes[-1]
        available = [edge_id for edge_id in edge_adjacency[node] if not visited[edge_id]]
        if available:
            previous_node = stack_nodes[-2] if len(stack_nodes) > 1 else None

            def continuation_score(edge_id: int) -> float:
                a, b = all_edges[edge_id]
                neighbor = b if a == node else a
                if previous_node is None:
                    return 0.0
                r0, c0 = coordinates[previous_node]
                r1, c1 = coordinates[node]
                r2, c2 = coordinates[neighbor]
                incoming = (r1 - r0, c1 - c0)
                outgoing = (r2 - r1, c2 - c1)
                denom = math.hypot(*incoming) * math.hypot(*outgoing) or 1.0
                return -(incoming[0] * outgoing[0] + incoming[1] * outgoing[1]) / denom

            edge_id = min(available, key=continuation_score)
            visited[edge_id] = 1
            first, second = all_edges[edge_id]
            neighbor = second if first == node else first
            stack_nodes.append(neighbor)
            stack_edges.append(edge_id)
        else:
            reverse_nodes.append(stack_nodes.pop())
            if stack_edges:
                reverse_edges.append(stack_edges.pop())

    if not all(visited):
        raise ValueError("The skeleton graph is disconnected after gap bridging")
    nodes = list(reversed(reverse_nodes))
    edge_ids = list(reversed(reverse_edges))
    segment_multiplicity = [1] * len(edge_ids)
    segment_widths = [all_edge_bridge_widths[edge_id] for edge_id in edge_ids]
    return nodes, edge_ids, segment_multiplicity, segment_widths, len(virtual_edges)


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
) -> str:
    if direction == "reverse":
        points = list(reversed(points))
        widths = list(reversed(widths))
        segment_multiplicity = list(reversed(segment_multiplicity))

    filament_area = math.pi * (FILAMENT_DIAMETER_MM * 0.5) ** 2
    travel_feed = int(round(TRAVEL_SPEED_MM_S * 60.0))
    lines = [
        f"; {name} — {direction} direction",
        "; Z is omitted: insert at the desired layer height.",
        "; Requires absolute XY (G90) and relative extrusion (M83) to be active.",
        f"; Speed target {PRINT_SPEED_MM_S:.2f} mm/s; flow capped at {MAX_FLOW_RATE_MM3_S:.3f} mm^3/s; bridges: {bridge_count}.",
        f"G0 X{points[0][0]:.3f} Y{points[0][1]:.3f} F{travel_feed}",
    ]

    for index in range(len(points) - 1):
        x0, y0 = points[index]
        x1, y1 = points[index + 1]
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 1e-6:
            continue
        width = max(MIN_EXTRUSION_WIDTH_MM, min(MAX_EXTRUSION_WIDTH_MM, (widths[index] + widths[index + 1]) * 0.5))
        area = bead_area(width, EXTRUSION_HEIGHT_MM)
        passes = max(1, segment_multiplicity[index])
        if passes > 1:
            shared_area = area / passes
            minimum_area = bead_area(MIN_EXTRUSION_WIDTH_MM, EXTRUSION_HEIGHT_MM)
            area = max(shared_area, minimum_area)
        speed = min(PRINT_SPEED_MM_S, MAX_FLOW_RATE_MM3_S * FLOW_SAFETY_FACTOR / area)
        e_delta = area * length / filament_area
        lines.append(
            f"G1 X{x1:.3f} Y{y1:.3f} E{e_delta:.5f} F{int(math.floor(speed * 60.0))}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"Input STL not found: {INPUT_FILE.resolve()}")
    if MIN_EXTRUSION_WIDTH_MM <= 0 or MAX_EXTRUSION_WIDTH_MM < MIN_EXTRUSION_WIDTH_MM:
        raise ValueError("Set positive widths with MAX_EXTRUSION_WIDTH_MM >= MIN_EXTRUSION_WIDTH_MM")
    if EXTRUSION_HEIGHT_MM <= 0 or MAX_FLOW_RATE_MM3_S <= 0 or FILAMENT_DIAMETER_MM <= 0:
        raise ValueError("Extrusion height, maximum flow, and filament diameter must be positive")
    if PRINT_SPEED_MM_S <= 0 or TRAVEL_SPEED_MM_S <= 0 or MAX_BRIDGE_GAP_MM <= 0:
        raise ValueError("Print/travel speed and maximum bridge gap must be positive")
    if not 0 < FLOW_SAFETY_FACTOR <= 1:
        raise ValueError("FLOW_SAFETY_FACTOR must be in (0, 1]")
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

    all_triangles = load_stl(INPUT_FILE)
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
    route, _edge_ids, segment_multiplicity, segment_width_overrides, topology_bridges = route_eulerian(
        coordinates,
        adjacency,
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

    base_name = INPUT_FILE.stem
    forward = build_gcode(base_name, points, widths, segment_multiplicity, bridge_count, "forward")
    reverse = build_gcode(base_name, points, widths, segment_multiplicity, bridge_count, "reverse")
    OUTPUT_FORWARD.write_text(forward, encoding="utf-8")
    OUTPUT_REVERSE.write_text(reverse, encoding="utf-8")

    print(f"Input facets: {len(all_triangles)}")
    print(f"Ignored facets/components: {ignored_count} facets ignored")
    print(f"Raster: {mask.shape[1]} x {mask.shape[0]} px at {RASTER_PIXELS_PER_MM:g} px/mm")
    print(f"Path points: {len(points)}; bridges added: {bridge_count}")
    print(f"Projected bounds: {u_max-u_min:.3f} x {v_max-v_min:.3f} mm")
    print(f"Wrote {OUTPUT_FORWARD.resolve()}")
    print(f"Wrote {OUTPUT_REVERSE.resolve()}")


if __name__ == "__main__":
    main()
