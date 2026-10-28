# lineart-printer
A small tool that converts single paths from vector graphics into gcode

## STL swept-line to variable-width G-code

`stl_to_variable_width_gcode.py` reads an STL directly and writes two Z-free
G-code snippets that follow the inferred centerline in opposite directions.
Edit the input/output paths and printer settings at the top of the script,
then run:

```sh
python3 stl_to_variable_width_gcode.py
```

The script requires NumPy (`python3 -m pip install numpy`). It projects the
mesh, filters small disconnected components (the pedestal in the supplied
STL), estimates a medial path, and joins nearby gaps up to
`MAX_BRIDGE_GAP_MM`. It computes E per distance from the rounded-rectangle
bead area and limits feedrate by `MAX_FLOW_RATE_MM3_S`.

The snippets assume absolute XY (`G90`) and relative extrusion (`M83`) are
already active. They do not change printer modes or include Z moves. The
default XY center is 110 mm, 110 mm; adjust it to align with the replacement
pedestal and your bed. The inferred centerline and automatic gap joins should
be inspected before printing.
