#####################################################################################
#####################################################################################
#########								    #########
#########								    #########
#########	          Scene filtering helper for easy use		    #########
#########			   version 1.0 by N4O			    #########
#########								    #########
#########								    #########
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
######### scene_filter(c=None, frame_num=None, maskimg=None, fn_filter=None, *args, **kwargs)
#########
######### c: clip
######### frame_num: frame that you want to scene filter (limited to 1 frame only, use int)
######### maskimg: Image Masking [Optional]
######### fn_filter: Function to use (example, core.f3kdb.Deband) <-- type without `()` or as string
######### *args: Argument for defined fn_filter
######### **kwargs: Keyword Argument for defined fn_filter
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

def scene_filter(c=None, frame_num=None, maskimg=None, fn_filter=None, *args, **kwargs):
	core = vs.get_core() # Get Core

	cbits = c.format.bits_per_sample # Check video bits
	imfam = str(maskimg.format.color_family)[12:]
	if frame_num is None:
		raise ValueError('scene_filter: frame_num cannot be empty')
	if isinstance(frame_num, int) or isinstance(frame_num, str):
		frame_num = int(frame_num)
	if c is None:
		raise ValueError('scene_filter: `c` cannot be empty (must be a clip)')
	elif not isinstance(c, vs.VideoNode):
		raise ValueError('scene_filter: `c` is not a clip')
	if frame_num >= c.num_frames:
		raise ValueError('scene_filter: frame_num cannot be more than \'{}\''.format(c.num_frames-1))
	if not cbits == 16:
		c = fvf.Depth(c, 16)

	if maskimg is None:
		im = core.imwri.Read(maskimg) # Read image
		usemask = True
	else:
		usemask = False

	if imfam != 'YUV' and usemask:
		im = core.resize.Spline36(clip=im, format=vs.YUV420P8, matrix_s="709") # Change to YUV if not YUV
		im = core.std.ShufflePlanes(im, 0, vs.GRAY) # ShufflePlanes to GRAYSCALE
		im = fvf.Depth(im, 16) # Dither to 16 bits
	elif imfam == 'YUV' and usemask:
		im = core.std.ShufflePlanes(im, 0, vs.GRAY) # ShufflePlanes to GRAYSCALE
		im = fvf.Depth(im, 16) # Dither to 16 bits
	else:
		pass
	
	def _inner_filter(n):
		if n < frame_num or n > frame_num:
			return c # Return normal clip if not in `frame_num` range
		ref = c # Set reference frames (not filtered)
		fil = fn_filter(ref, *args, **kwargs) # Filter frame
		if usemask:
			fil = core.std.MaskedMerge(fil, ref, im) # Use mask if applied
		return fil # Return filtered frame_num

	return core.std.FrameEval(c, _inner_filter) # Return 16 bits video
