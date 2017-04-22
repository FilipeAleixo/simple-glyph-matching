# simple-glyph-matching

## Synopsys

This code is a simple, yet efficient, implementation for glyph matching. 

It is useful, for instance, if you have a large set of glyphs (here refered to as *glyphs to process*) that you want to match to a reference set of glyphs (here referred to as *model glyphs*). 

In order for the matching to be successful, there must be a reasonably high similarity between the glyphs to process and the model glyphs, since matching is simply done by pixel superimposition.

This code is applicable to large quantities of data, as it relies on simple calculations.

## How to use

The model glyphs are contained in 'MODELS', while the glyphs to process are contained in the different folders under 'TO_PROCESS'.
By calling main.py, the script iterates throughout all the folders under 'TO_PROCESS' and creates a .csv file with the same name as the folder, which contains the matches between the glyphs to process and the model glyphs.