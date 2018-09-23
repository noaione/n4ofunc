#####################################################################################
#####################################################################################
#########                                                                   #########
#########                                                                   #########
#########               Scene filtering helper for easy use                 #########
#########                        version 1.1 by N4O                         #########
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
######### mappings: Frame Mappings, can be a single frame (int or str) and multiple frames (tuple or list)
######### maskimg: Use a specified mask image to limit filtering area (Can be used or not)
######### fn_filter: Filter function to filter the clip provided, example: `core.f3kdb.Deband` (just put the function name)
######### filter_args: To adjust specified filter setting, for example for the f3kdb.Deband filter: `[12, 60, 40, 40]` (same as: range 12, y 60, cb/cr 40)
######### filter_kwargs: Same as `filter_args` but using dict-type, for example: `{'range': 12, 'y': 60, 'cb': 40, 'cr': 40}`
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


def scene_filter(clip=None, mappings=None, maskimg=None, fn_filter=None, filter_args=[], filter_kwargs={}):
	cbits = clip.format.bits_per_sample # Check video bits
	if isinstance(mappings, int) or isinstance(mappings, str):
		frame_collection = [int(mappings)]
	elif isinstance(mappings, list) or isinstance(mappings, tuple):
		frame_collection = []
		for frame in mappings:
			frame_collection.append(int(frame))
	else:
		raise ValueError('scene_filter: mappings can only be a single frame (integer) or multiple frame (list or tuple)')
	if clip is None:
		raise ValueError('scene_filter: `c` cannot be empty (must be a clip)')
	elif not isinstance(clip, vs.VideoNode):
		raise ValueError('scene_filter: `c` is not a clip')
	if clip.num_frames - 1 in mappings:
		raise ValueError('scene_filter: mappings cannot be more than \'{}\''.format(clip.num_frames-1))
	if cbits != 16:
		clip = fvf.Depth(clip, 16)

	if maskimg is not None:
		im = core.imwri.Read(maskimg) # Read image
		usemask = True
	else:
		usemask = False

	if usemask:
		im = core.resize.Spline36(clip=im, format=vs.GRAYS, matrix_s="709") # Change to GRAYS
		im = fvf.Depth(im, 16) # Dither to 16 bits
	else:
		pass
	
	def filter_frame(n, clp):
		if n not in mappings:
			return clp # Return normal clip if not in `mappings` range
		ref = clp # Set reference frames (not filtered)
		fil = fn_filter(ref, *filter_args, **filter_kwargs) # Filter frame
		if usemask:
			fil = core.std.MaskedMerge(fil, ref, im) # Use mask if applied
		return fil # Return filtered frame

	return core.std.FrameEval(clip, functools.partial(filter_frame, clp=clip)) # Return 16 bits video