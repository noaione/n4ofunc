# n4ofunc

A collection of VapourSynth script that I used sometimes.

## Requirements
1. VapourSynth R51+
2. [fvsfunc](https://github.com/Irrational-Encoding-Wizardry/fvsfunc/blob/master/fvsfunc.py)
3. [vsutil](https://github.com/Irrational-Encoding-Wizardry/vsutil/blob/master/vsutil.py)

**Optional**:
1. [havsfunc](https://github.com/HomeOfVapourSynthEvolution/havsfunc/blob/master/havsfunc.py) `Used at adaptive_degrain2 (SMDegrain)`
2. [mvsfunc](https://github.com/HomeOfVapourSynthEvolution/mvsfunc/blob/master/mvsfunc.py) `Used at adaptive_degrain2 (BM3D)`

## Functions
### best.py
A collection of script to get the best possible quality from multi-source video.

1. better_frame
2. better_planes

### comp.py
A collection of comparision script

1. check_difference
2. save_difference
3. stack_compare
   - interleave_compare
   - compare

`check_difference` is mostly the same as `save_difference`, but instead of saving the frame difference into a picture `check_difference` would only print out into console what frame have difference from the provided threshold.

`stack_compare` or `compare` is a way to compare multiple video or clip and `interleave_compare` is a wrapped partial function for stack_compare that interleave between multiple video

### degrain.py
A collection of degraining script.

1. adaptive_degrain2
   - adaptive_bm3d
   - adaptive_dfttest
   - adaptive_knlm
   - adaptive_tnlm
   - adaptive_smdegrain

This module only contains a single actual function which is `adaptive_degrain2`, everything else is a partial wrapped function for `adaptive_degrain2` where the kernel is changed.

### mask.py
A collection of mask creation tools.

1. antiedgemask
   - antiedge
2. simple_native_mask
   - native_mask
3. recursive_apply_mask
   - rapplym

### scale.py
A collection of scaling function.

1. masked_descale
2. upscale_nnedi3
3. adaptive_scaling
   - adaptive_rescale
   - adaptive_descale

`upscale_nnedi3` is actually a typed-version and "fixed" version of `nnedi3_rpow2`. The reason is because the latest version of Vapoursynth does not like it.

### utils.py
A collection of utilities script, might not be useful.

1. is_extension
2. register_format

### video.py
A collection of script to handle video or image source.

1. source
   - src
2. SimpleFrameReplace
   - frame_replace
   - sfr
3. debug_clip
   - debugclip
4. shift_444

More information of every script can be seen from the docstring.

## License

This project is licensed with [MIT License](LICENSE).
