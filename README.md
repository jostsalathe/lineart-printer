# lineart-printer

Convert a swept-line STL into alternating variable-width paths and insert them
into selected layers of an existing PrusaSlicer ASCII G-code print.

## todos

- The resulting tool path has jagged sections. Some smoothing should be integrated.
- The whole intput and output selection could be moved to the web GUI.

## STL swept-line to G-code

Edit the input paths, layer range, maximum width, and XY center at the top of
`stl_to_variable_width_gcode.py`, then run:

```sh
python3 stl_to_variable_width_gcode.py
```

The script requires NumPy:

```sh
python3 -m pip install numpy
```

Crossing review opens a local browser page. The tool reads and writes ASCII
`.gcode` files directly and leaves the input file unchanged.

`INPUT_STL` supplies the line geometry. The script projects the mesh, ignores
small disconnected components such as the integrated pedestal, skeletonizes
the silhouette, and opens a browser review for each crossing. Nearby odd-arm
skeleton junctions are grouped within `MAX_CROSSING_CLUSTER_SPAN_MM` so each
crossing is presented with an even number of arms. Click two colored arm tips
to say which line ends continue through that crossing. Nearby gaps up to
`MAX_BRIDGE_GAP_MM` can be joined automatically. After reviewing all crossings,
the script verifies that the selected connections make one continuous path.
If they form multiple trails, the browser review stays open so you can revisit
any crossing; its previous pairings are restored for editing. A colored
overview is shown before completion, with a different color and start/end
markers for each trail. Click numbered crossings in the graphic to edit them. The route is
checked automatically after the final crossing and whenever an edited crossing
is saved; a valid overview can then be confirmed to continue.

`INPUT_GCODE_FILE` supplies the host print, including the replacement pedestal;
`OUTPUT_GCODE_FILE` receives the modified ASCII G-code.
Set `START_LAYER` to the first layer where the line should print and
`LAYER_COUNT` to the number of consecutive layers. The first inserted path runs
forward, the next runs in reverse, and the directions alternate. Layer heights
and the active `M221` flow multiplier are read from the host G-code. E amounts
are adjusted for that multiplier. When set to `None`, the minimum width comes
from PrusaSlicer's external perimeter width, maximum flow comes from the lowest
positive material or print volumetric limit, print speed comes from the
external perimeter speed, and travel speed comes from the travel speed. Set a
numeric value at the top of the script to override any of these. These values
are read from PrusaSlicer's configuration comments in the ASCII G-code. Filament
diameter is read from the G-code comments where available; otherwise the
1.75 mm fallback or the explicit `FILAMENT_DIAMETER_MM_OVERRIDE` can be used.
Automatic extraction from ASCII G-code requires the PrusaSlicer config comments
to be present in the file; set any missing values explicitly at the top of the
script.

PrusaSlicer does not export a standard flow safety factor setting. The script
keeps its own `FLOW_SAFETY_FACTOR` margin (0.98 by default); it can also read a
custom `flow_safety_factor` G-code comment if one is present.

Forward paths are inserted at the end of their layer, after its existing
toolpaths and before the next layer marker. Reverse paths are inserted at the
beginning of the following layer, after its Z setup and priming but before its
first XY extrusion. This makes each forward/reverse pair directly straddle a
layer change. If a layer has no existing XY extrusion, its beginning insertion
falls back to the end of that layer's setup section.
Each complete pair ends with a non-extruding travel back to the XY position
where the forward injection began, then restores the host layer's original XY
resume position. `LAYER_COUNT` must be even so every injection is a complete pair.

The generated routes use absolute XY (`G90`) and relative extrusion (`M83`),
omit Z moves, and are marked in the G-code so rerunning the script replaces
its previous generated paths. Set `OUTPUT_CENTER_X_MM` and
`OUTPUT_CENTER_Y_MM` to align the route with the replacement pedestal. The
input host should contain the pedestal but not another copy of the line-art
toolpath; otherwise that original path would also print on the selected
layers. Inspect the resulting G-code in a slicer or G-code viewer before
printing.
