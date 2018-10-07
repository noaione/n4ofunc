### N4O Function for Vapoursynth

How to install:
```py
1. Download/Clone this repo
2. Extract the files
3. Copy n4ofunc.py to \Python36\Lib\site-packages
4. Load it, ex: import n4ofunc as nao
```

(stolen) Feature:
- **IVTC, Decimate, Deinterlace, etc..**
- **hdr2sdr (experimental)** [Hable Tonemapping Algorithm, should now provide better result] [Aliases: tonemap, hdr2sdr]
- **descale_rescale** [Aliases: rescale]
- **Hybrid denoise** (S_hybriddenoise, M_hybriddenoise, H_hybriddenoise)
- **mpeg2stinx** (Ported from avs function)

Version rev:
- Version 1.0: first commit
- Version 1.1: Fix autistCode, added a little bit option for High SMDegrain
- Version 1.2: Some changes for High Deband, cleaned up script. Changed argument for i444
- Version 1.3: Added descale_rescale, reworked autistCode and renamed it to benchmark
- Version 1.3.1: Fixed KNLMeansCL, some tweak and clean up
- Version 1.4: Reworked hdr2sdr, now using hable tonemap algorithm that stolen from age@Doom9 with some additional command. Added some aliases
- Version 1.5: Removed useless shit, moved my mpeg2stinx from different repo to this script


### scenefilter.py

```py
import scenefilter as sfn

v = core.ffms2.Source('clip.mkv')
v = sfn.scene_filter(v, "30", 'mask_frame30.png', core.f3kdb.Deband, [12, 60, 40, 40], {'grainy': 15, 'grainc': 0, 'output_depth': 16}) 
### Deband with range 12, y 60, cb/cr 40, grainy 15, grainc 0, and output_depth 10 for frame 30 with mask

v.set_output()
```

Supported mask:
- b/w image mask
- .ass mask

**Parameters**:
::
    *clip*
        Vapoursynth clip (Source clip)
    *mappings*
        Frame Mappings (string)
    *mask*
        Use a specified mask image to limit filtering area (Can be used or not)
    *fn_filter*
        Filter function to filter the clip provided, example: `core.f3kdb.Deband` (just put the function name)
    *filter_args*
        To adjust specified filter setting, for example for the f3kdb.Deband filter: `[12, 60, 40, 40]` (same as: range 12, y 60, cb/cr 40)
    *filter_kwargs* 
        Same as `filter_args` but using dict-type, for example: `{'range': 12, 'y': 60, 'cb': 40, 'cr': 40}`

Each line in the tmappings string must have one of the following forms:

- a z

    Filter only frame `a` and frame `z`

- [a b] z

    Filter all frames in the range `[a b]` and filter frame `z`

- [a b] [y z]

    Filter all frames in the range `[a b]` and filter frames in the range `[y z]`

Example:
::
    [0 9] [12 14]   # Filter frame 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14
    10 5            # Filter frame 10 & 5
    [15 20] 6       # Filter frame 15, 16, 17, 18, 19, 20 and frame 6
Within each line, all whitespace is ignored.

Version rev:
- Version 1.0: First commit
- Version 1.1: Code improvement and some fixes, and now support multiple frame
- Version 1.2: Now support .ass file mask, fixed wrong MaskedMerge argument, 
  Added `import` things because i'm dumb enough to forget it