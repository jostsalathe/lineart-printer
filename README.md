# lineart-printer

Convert a swept-line STL into alternating variable-width paths and insert them
into selected layers of an existing PrusaSlicer G-code or BGCode print.

## STL swept-line to G-code/BGCode

Edit the input paths, layer range, maximum width, and XY center at the top of
`stl_to_variable_width_gcode.py`, then run:

```sh
python3 stl_to_variable_width_gcode.py
```

The script requires NumPy. Prusa's BGCode Python bindings are only needed
when either configured print file has a `.bgcode` extension. Install NumPy,
then build/install the bindings from Prusa's official source repository if you
want BGCode support:

```sh
python3 -m pip install numpy
git clone https://github.com/prusa3d/libbgcode.git
cd libbgcode
python3 -m pip install ./
```

On some Linux systems, Tkinter also needs the `python3-tk` system package. Set
`INPUT_PRINT_FILE` and `OUTPUT_PRINT_FILE` to `.gcode` for ASCII G-code or
`.bgcode` for BGCode. The script preserves the configured output format and
leaves the input file unchanged. BGCode is decoded and encoded with Prusa's
library. ASCII G-code can be processed directly without that library.

`INPUT_STL` supplies the line geometry. The script projects the mesh, ignores
small disconnected components such as the integrated pedestal, skeletonizes
the silhouette, and opens a graphic review for each crossing. Click two
colored arm tips to say which line ends continue through that crossing. Nearby
gaps up to `MAX_BRIDGE_GAP_MM` can be joined automatically.

`INPUT_PRINT_FILE` supplies the host print, including the replacement pedestal.
Set `START_LAYER` to the first layer where the line should print and
`LAYER_COUNT` to the number of consecutive layers. The first inserted path runs
forward, the next runs in reverse, and the directions alternate. Layer heights
and the active `M221` flow multiplier are read from the host G-code. E amounts
are adjusted for that multiplier. When set to `None`, the minimum width comes
from PrusaSlicer's external perimeter width, maximum flow comes from the lowest
positive material or print volumetric limit, print speed comes from the
external perimeter speed, and travel speed comes from the travel speed. Set a
numeric value at the top of the script to override any of these. These values
are read from PrusaSlicer's configuration comments in ASCII G-code, or from
BGCode metadata/comments when available. Filament diameter is read from
BGCode/G-code metadata where available; otherwise the 1.75 mm fallback or the
explicit `FILAMENT_DIAMETER_MM_OVERRIDE` can be used.
Automatic extraction from ASCII G-code requires the PrusaSlicer config comments
to be present in the file; set any missing values explicitly at the top of the
script.

PrusaSlicer does not export a standard flow safety factor setting. The script
keeps its own `FLOW_SAFETY_FACTOR` margin (0.98 by default); it can also read a
custom `flow_safety_factor` metadata comment if one is present.

Set `INSERTION_POINT` to `"beginning"` or `"end"`. The beginning option inserts
after the layer's Z setup and priming, just before its first XY extrusion move.
The end option inserts after that layer's existing toolpaths and before the next
layer marker.

The generated routes use absolute XY (`G90`) and relative extrusion (`M83`),
omit Z moves, and are marked in the G-code so rerunning the script replaces
its previous generated paths. Set `OUTPUT_CENTER_X_MM` and
`OUTPUT_CENTER_Y_MM` to align the route with the replacement pedestal. The
input host should contain the pedestal but not another copy of the line-art
toolpath; otherwise that original path would also print on the selected
layers. Inspect the resulting G-code/BGCode in a slicer or G-code viewer before
printing.
