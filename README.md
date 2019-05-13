### N4O Function for Vapoursynth

A simple function collection that I used

**Requirements**:
1. [nnedi3_rpow2](https://github.com/darealshinji/vapoursynth-plugins/blob/master/scripts/nnedi3_rpow2.py)
2. [fvsfunc](https://github.com/Irrational-Encoding-Wizardry/fvsfunc/blob/master/fvsfunc.py)
3. [havsfunc](https://github.com/HomeOfVapourSynthEvolution/havsfunc/blob/master/havsfunc.py)
4. [kagefunc](https://github.com/Irrational-Encoding-Wizardry/kagefunc/blob/master/kagefunc.py)
5. [mvsfunc](https://github.com/HomeOfVapourSynthEvolution/mvsfunc/blob/master/mvsfunc.py)
6. [MaskDetail](https://gist.github.com/noaione/4d89940d52b5bf33f7d685825c88f4f2)
7. [vsutil](https://github.com/Irrational-Encoding-Wizardry/vsutil/blob/master/vsutil.py)


### scenefilter.py

```py
import scenefilter as sfn

v = core.ffms2.Source('clip.mkv')
v = sfn.scene_filter(v, "30 [10 15]", 'mask_frame30.png', core.f3kdb.Deband, [12, 60, 40, 40], {'grainy': 15, 'grainc': 0, 'output_depth': 16}) 
### Deband with range 12, y 60, cb/cr 40, grainy 15, grainc 0, and output_depth 10 for frame 30 and from frame 10 to frame 15 with mask

v.set_output()
```

**Supported mask**:
- B/W image mask
- .ass mask

**Parameters:**
- ***clip***
    >Vapoursynth clip (Source clip)
- ***mappings***
    >Frame Mappings (string)
- ***mask***
    >Use a specified mask image to limit filtering area (Can be used or not)
- ***fn_filter***
    >Filter function to filter the clip provided, example: `core.f3kdb.Deband` (just put the function name)
- ***filter_args***
    >To adjust specified filter setting, for example for the f3kdb.Deband filter: `[12, 60, 40, 40]` (same as: range 12, y 60, cb/cr 40)
- ***filter_kwargs***
    >Same as `filter_args` but using dict-type, for example: `{'range': 12, 'y': 60, 'cb': 40, 'cr': 40}`

Each line in the tmappings string must have one of the following forms:

- a z
    - Filter only frame `a` and frame `z`

- [a b] z
    - Filter all frames in the range `[a b]` and filter frame `z`

- [a b] [y z]
    - Filter all frames in the range `[a b]` and filter frames in the range `[y z]`

Example:
- [0 9] [12 14]   `# Filter frame 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14`
- 10 5            `# Filter frame 10 & 5`
- [15 20] 6       `# Filter frame 15, 16, 17, 18, 19, 20 and frame 6`
- Within each line, all whitespace is ignored.

Version rev:
- Version 1.0: First commit
- Version 1.1: Code improvement and some fixes, and now support multiple frame
- Version 1.2: Now support .ass file mask, fixed wrong MaskedMerge argument, 
  Added `import` things because i'm dumb enough to forget it
