#####################################################################################
#####################################################################################
#########                                                                   #########
#########                                                                   #########
#########               Scene filtering helper for easy use                 #########
#########                        version 1.2 by N4O                         #########
#########                                                                   #########
#########                                                                   #########
#####################################################################################
#####################################################################################
#########
######### Example:
######### ```
######### import vapoursynth as vs 
######### import scenefilter as sfn
######### 
######### v = core.ffm2.Source('test.mkv')
######### v = sfn.scene_fiiler(v, 400, 'mask1.png', core.grain.Add, ['2', '2']) # <-- Add grain hcor 2 vcor 2 to frame 400 of v with b/w mask
######### v.set_output()
######### ```
#########
#####################################################################################
#####################################################################################
#########
######### Explanation:
######### scene_filter(clip=None, mappings=None, maskimg=None, fn_filter=None, filter_args=[], filter_kwargs={})
#########
######### clip: Vapoursynth clip
######### mappings: Frame Mappings (string)
######### maskimg: Use a specified mask image to limit filtering area (Can be used or not)
######### fn_filter: Filter function to filter the clip provided, example: `core.f3kdb.Deband` (just put the function name)
######### filter_args: To adjust specified filter setting, for example for the f3kdb.Deband filter: `[12, 60, 40, 40]` (same as: range 12, y 60, cb/cr 40)
######### filter_kwargs: Same as `filter_args` but using dict-type, for example: `{'range': 12, 'y': 60, 'cb': 40, 'cr': 40}`
#########
#####################################################################################
#####################################################################################
#########
######### Frame mappings explanation:
######### 
######### - a z
#########     Filter only frame `a` and frame `z`
######### 
######### - [a b] z
#########     Filter all frames in the range `[a b]` and filter frame `z`
######### 
######### - [a b] [y z]
#########     Filter all frames in the range `[a b]` and filter frames in the range `[y z]`
######### 
######### Example:
######### ::
#########     [0 9] [12 14]   # Filter frame 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14
#########     10 5            # Filter frame 10 & 5
#########     [15 20] 6       # Filter frame 15, 16, 17, 18, 19, 20 and frame 6
######### Within each line, all whitespace is ignored.
#########
#####################################################################################
#####################################################################################
#########
######### Dependencies:
######### - VapourSynth
######### - Python3 64bit
######### - fvsfunc [https://github.com/Irrational-Encoding-Wizardry/fvsfunc]
#########
#####################################################################################
#####################################################################################

import vapoursynth as vs
import fvsfunc as fvf
import functools
import re

core = vs.get_core()

def scene_filter(clip=None, mappings=None, mask=None, fn_filter=None, filter_args=[], filter_kwargs={}):
	if clip is None:
		raise ValueError('scene_filter: `c` cannot be empty (must be a clip)')
	elif not isinstance(clip, vs.VideoNode):
		raise ValueError('scene_filter: `c` is not a clip')

	cbits = clip.format.bits_per_sample # Check video bits
	cframes = clip.num_frames
	if mask is not None:
		maskfmt = mask[mask.rfind('.'):len(mask)]

	if cbits != 16:
		clip = fvf.Depth(clip, 16)

	if mappings is not None:
		if not isinstance(mappings, str):
			raise ValueError('scene_filter: `mappings` must be a string')

		_mappings = mappings.replace(',', ' ').replace(':', ' ')
		frames = re.findall(r'\d+(?!\d*\s*\d*\s*\d*\])', _mappings) ### this part shamelessly stolen from fvsfun
		ranges = re.findall(r'\[\s*\d+\s+\d+\s*\]', _mappings) ### also this part
		mappings = []
		for range_ in ranges:
			_r = range_.strip('[ ]').split()
			mappings.extend(list(range(int(_r[0]), int(_r[1])+1)))
		for frame in frames:
			mappings.append(int(frame))

		for frame in mappings: # Error checking if there is frame mapped larger than 
			if frame >= cframes:
				raise ValueError('scene_filter: Frame cannot be larger than: {} (Detected frame: {})'.format(cframes, frame))

	if mask is not None and maskfmt != '.ass':
		im = core.imwri.Read(mask) # Read image
		im = core.std.ShufflePlanes(im, 1, colorfamily=vs.GRAY) # Change to GRAYS
		im = fvf.Depth(im, 16) # Dither to 16 bits
		im = core.std.BoxBlur(im, hradius=5, vradius=5) # Blur the edge
		usemask = True
	elif mask is not None and maskfmt == '.ass':
		bclip = core.std.BlankClip(width=clip.width, height=clip.height, length=cframes) # Create blank black clip
		im = core.sub.TextFile(bclip, mask).std.Binarize() # Merge .ass mask to the blank clip
		im = core.std.ShufflePlanes(im, 1, colorfamily=vs.GRAY) # Change to GRAYS
		im = fvf.Depth(im, 16) # Dither to 16 bits
		im = core.std.BoxBlur(im, hradius=5, vradius=5) # Blur the edge
		usemask = True
	else:
		usemask = False
	
	def filter_frame(n, c):
		if n not in mappings:
			return c # Return normal clip if not in `mappings` range
		ref = c # Set reference frames (not filtered)
		fil = fn_filter(ref, *filter_args, **filter_kwargs) # Filter frame
		if usemask:
			fil = core.std.MaskedMerge(ref, fil, im) # Use mask if applied
		return fil # Return filtered frame

	if maskfmt == '.ass':
		ref = clip
		fil = fn_filter(ref, *filter_args, **filter_kwargs)
		return core.std.MaskedMerge(ref, fil, im)
	return core.std.FrameEval(clip, functools.partial(filter_frame, c=clip)) # Return 16 bits video