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

def scene_filter(clip=None, mappings=None, mask=None, fn_filter=None, filter_args=None, filter_kwargs=None):
	core = vs.core # Init
	class SceneFilterException(Exception): # Custom Exception
		__module__ = Exception.__module__
		
	if not isinstance(clip, vs.VideoNode):
		raise SceneFilterException('`clip` must be a clip (vapoursynth.VideoNode)')
	
	# Set default
	if fn_filter is None:
		raise SceneFilterException('`fn_filter` cannot be empty')
	if filter_args is None:
		filter_args = []
	if filter_kwargs is None:
		filter_kwargs = {}
	maskfmt = None # Set as None so it doesn't broke
	cbits = clip.format.bits_per_sample # Check video bits
	cframes = clip.num_frames # Check total frames
	
	if mask is not None:
		maskfmt = mask[mask.rfind('.'):len(mask)]
	if cbits != 16:
		clip = fvf.Depth(clip, 16)

	if mappings is not None:
		if not isinstance(mappings, str):
			raise SceneFilterException('`mappings` must be a string')

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
		sk = core.imwri.Read(mask) # Read image
		sk = core.std.ShufflePlanes(sk, 1, colorfamily=vs.GRAY) # Change to GRAYS
		sk = fvf.Depth(sk, 16) # Dither to 16 bits
		sk = core.std.BoxBlur(sk, hradius=5, vradius=5) # Blur the edge
	elif mask is not None and maskfmt == '.ass':
		bclip = core.std.BlankClip(width=clip.width, height=clip.height, length=cframes) # Create blank black clip
		sk = core.sub.TextFile(bclip, mask).std.Binarize() # Merge .ass mask to the blank clip
		sk = core.std.ShufflePlanes(sk, 1, colorfamily=vs.GRAY) # Change to GRAYS
		sk = fvf.Depth(sk, 16) # Dither to 16 bits
		sk = core.std.BoxBlur(sk, hradius=5, vradius=5) # Blur the edge
	else:
		sk = None
	
	def filter_frame(n, c, _fn, _fargs, _fkwargs, ma):
		if n not in mappings:
			return c # Return normal clip if not in `mappings` range
		ref = c # Set reference frames (not filtered)
		try:
			fil = _fn(ref, *_fargs, **_fkwargs) # Filter frame
		except vs.Error as err:
			raise SceneFilterException('filter_frame: `fn_filter` defined not found or args and kwargs are undefined\nOriginal Exception: {}'.format(err))
		if ma:
			fil = core.std.MaskedMerge(ref, fil, ma) # Use mask if applied
		return fil # Return filtered frame

	if maskfmt == '.ass':
		ref = clip
		try:
			fil = fn_filter(ref, *filter_args, **filter_kwargs) # Filter with mask
		except vs.Error as err:
			raise SceneFilterException('filter_frame: `fn_filter` defined not found or args and kwargs are undefined\nOriginal Exception: {}'.format(err))
		return core.std.MaskedMerge(ref, fil, sk)
	return core.std.FrameEval(clip, functools.partial(filter_frame, c=clip, _fn=fn_filter, _fargs=filter_args, _fkwargs=filter_kwargs, ma=sk))