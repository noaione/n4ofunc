import vapoursynth as vs
import fvsfunc as fvf
import havsfunc as haf
import mvsfunc as mvf
import kagefunc as kgf
import edi_rpow2 as edi
from functools import partial

#init
core = vs.core

#helper
def to_plane_array(clip):
	"""
	Stolen from Kageru function
	"""
	return [core.std.ShufflePlanes(clip, x, colorfamily=vs.GRAY) for x in range(clip.format.num_planes)]
	
def iterate(vid, filter, amount):
	if amount == 0:
		return vid
	for _ in range(amount):
		vid = filter(vid)
	return vid
	
def splitYUV(src):
	return [core.std.ShufflePlanes(src, x, colorfamily=vs.GRAY) for x in range(src.format.num_planes)]
	
def getY(src):
	return core.std.ShufflePlanes(src, 0, vs.GRAY)

###################################################################
# Dehalo filter based on Dehalo_alpha and custom simple mask
###################################################################
# rx, ry [float, 1.0 ... 2.0 ... ~3.0]
# As usual, the radii for halo removal.
# Note: this function is rather sensitive to the radius settings. Set it as low as possible! If radius is set too high, it will start missing small spots.
#
# darkkstr, brightstr [float, 0.0 ... 1.0] [<0.0 and >1.0 possible]
# The strength factors for processing dark and bright halos. Default 1.0 both for symmetrical processing.
# On Comic/Anime, darkstr=0.4~0.8 sometimes might be better ... sometimes. In General, the function seems to preserve dark lines rather good.
#
# lowsens, highsens [int, 0 ... 50 ... 100]
# Sensitivity settings, not that easy to describe them exactly ...
# In a sense, they define a window between how weak an achieved effect has to be to get fully accepted, and how strong an achieved effect has to be to get fully discarded.
# Defaults are 50 and 50 ... try and see for yourself.
#
# ss [float, 1.0 ... 1.5 ...]
# Supersampling factor, to avoid creation of aliasing.
#
# thr [int, 1 ... 65535]
# Maximum threshold factor
# Recommended: 15000+
# Max: Clip Sample Value (65535)
#
def nDeHalo(clp=None, rx=None, ry=None, darkstr=None, brightstr=None, lowsens=None, highsens=None, ss=None, thr=None):
	import math

	def m4(x):
		return 16 if x < 16 else math.floor(x / 4 + 0.5) * 4
	def scale(value, peak):
		return value * peak // 255

	# Defaults
	rx = 2. if rx is None else rx
	ry = 2. if ry is None else ry
	darkstr = 1. if darkstr is None else darkstr
	brightstr = 1. if brightstr is None else brightstr
	lowsens = 50 if lowsens is None else lowsens
	highsens = 50 if highsens is None else highsens
	ss = 1.5 if ss is None else ss
	thr = 65535 if thr is None else thr

	# Error Check
	if not isinstance(clp, vs.VideoNode):
		raise ValueError("nDeHalo: This is not a clip")
	if thr > 65535:
		raise ValueError("nDeHalo: thr cannot exceed 65535: {x}".format(x=thr))
	
	# Mask creation
	sx = clp.std.Convolution([-1, -2, -1, 0, 0, 0, 1, 2, 1], saturate=False)
	sy = clp.std.Convolution([-1, 0, 1, -2, 0, 2, -1, 0, 1], saturate=False)
	inner = core.std.Expr([sx, sy], 'x y max').std.ShufflePlanes(0, vs.GRAY)
	outer = inner.std.Maximum(threshold=thr)
	mask = core.std.Expr([outer, inner], 'x y -')
	
	peak = (1 << clp.format.bits_per_sample) - 1
	
	# Initial Check
	if clp.format.color_family != vs.GRAY:
		clp_orig = clp
		clp = core.std.ShufflePlanes(clp, 0, vs.GRAY)
	else:
		clp_orig = None
	
	ox = clp.width
	oy = clp.height
	
	# DeHalo
	halos = core.resize.Bicubic(clp, m4(ox / rx), m4(oy / ry)).resize.Bicubic(ox, oy, filter_param_a=1, filter_param_b=0)
	are = core.std.Expr([core.std.Maximum(clp), core.std.Minimum(clp)], ['x y -'])
	ugly = core.std.Expr([core.std.Maximum(halos), core.std.Minimum(halos)], ['x y -'])
	expr = 'y x - y / {peak} * {LOS} - y {i} + {j} / {HIS} + *'.format(peak=peak, LOS=scale(lowsens, peak), i=scale(256, peak), j=scale(512, peak), HIS=highsens / 100)
	so = core.std.Expr([ugly, are], [expr])
	lets = core.std.MaskedMerge(halos, clp, so)
	if ss <= 1:
		remove = core.rgvs.Repair(clp, lets, 1)
	else:
		remove = core.std.Expr([core.std.Expr([core.resize.Spline36(clp, m4(ox * ss), m4(oy * ss)),
                                               core.std.Maximum(lets).resize.Bicubic(m4(ox * ss), m4(oy * ss))],
                                              ['x y min']),
                                core.std.Minimum(lets).resize.Bicubic(m4(ox * ss), m4(oy * ss))],
                               ['x y max']).resize.Spline36(ox, oy)
	them = core.std.Expr([clp, remove], ['x y < x x y - {DRK} * - x x y - {BRT} * - ?'.format(DRK=darkstr, BRT=brightstr)])
	
	# Merge
	if clp_orig is not None:
		final = core.std.ShufflePlanes([them, clp_orig], planes=[0, 1, 2], colorfamily=clp_orig.format.color_family)
		fc = clp_orig
	else:
		final = them
		fc = clp
	return core.std.MaskedMerge(fc, final, mask)

def S_hybriddenoise(src, knl=0.4, tr=2, thsad=50, thsadc=75):
	"""
	src: video input
	knl or h: knlmeanscl
	tr, thsad, thsadc: smdegrain
	##Stolen from kageru function
	using smdegrain for denoise luma and knlmeanscl for denoise chroma
	"""
	planes = to_plane_array(src)
	planes[0] = haf.SMDegrain(planes[0], tr=tr, thSAD=thsad, thSADC=thsadc, prefilter=3, RefineMotion=True, search=4)
	planes[1], planes[2] = [core.knlm.KNLMeansCL(plane, a=2, h=knl, d=3, s=8, device_type='gpu', device_id=0)
							for plane in planes[1:]]
	return core.std.ShufflePlanes(clips=planes, planes=[0, 0, 0], colorfamily=vs.YUV)

def M_hybriddenoise(src, radius1=1, sigma=2, tr=2, thsad=50, thsadc=75):
	"""
	src: video input
	radius1 & sigma: bm3d
	tr, thsad, thsadc: smdegrain
	##Stolen from kageru function
	using bm3d for denoise luma and smdegrain for denoise chroma
	"""
	planes = to_plane_array(src)
	planes[0] = haf.SMDegrain(planes[0], tr=tr, thSAD=thsad, thSADC=thsadc, prefilter=3, RefineMotion=True, search=4)
	planes[1], planes[2] = [mvf.BM3D(plane, radius1=radius1, sigma=sigma)
							for plane in planes[1:]]
	return core.std.ShufflePlanes(clips=planes, planes=[0, 0, 0], colorfamily=vs.YUV)

def H_hybriddenoise(src, knl=0.4, sigma=2, radius1=1):
	"""
	src: video input
	knl: is 
	##Stolen from kageru function
	using bm3d for denoise luma and knlmeanscl for denoise chroma
	"""
	planes = to_plane_array(src)
	planes[0] = mvf.BM3D(planes[0], radius1=radius1, sigma=sigma)
	planes[1], planes[2] = [core.knlm.KNLMeansCL(plane, a=2, h=knl, d=3, s=8, device_type='gpu', device_id=0)
							for plane in planes[1:]]
	return core.std.ShufflePlanes(clips=planes, planes=[0, 0, 0], colorfamily=vs.YUV)
	
def adaptive_smdegrain(src, thSAD=None, thSADC=None, luma_scaling=None, area='light', show_mask=False):
	if thSAD is None:
		thSAD = 150
	if thSADC is None:
		thSADC = thSAD
	
	if luma_scaling is None:
		luma_scaling = 30
	
	if area != 'light' and area != 'dark':
		raise ValueError('n4ofunc.adaptive_smdegrain: `area` can only be: `light` and `dark`')
	
	adaptmask = kgf.adaptive_grain(src, luma_scaling=luma_scaling, show_mask=True)
	
	Yplane = getY(src)
	
	if area == 'light':
		adaptmask = adaptmask.std.Invert()
		
	limitx = Yplane.std.Convolution([-1, -2, -1, 0, 0, 0, 1, 2, 1], saturate=False)
	limity = Yplane.std.Convolution([-1, 0, 1, -2, 0, 2, -1, 0, 1], saturate=False)
	limit = core.std.Expr([limitx, limity], 'x y max')
	
	mask = core.std.Expr([adaptmask, limit], 'x y -')

	if show_mask:
		return mask

	fil = haf.SMDegrain(src, thSAD=thSAD, thSADC=thSADC)

	return core.std.MaskedMerge(src, fil, mask)
	
def antiedgemask(src, iter=1):
	w = src.width
	h = src.height
	
	Yplane = getY(src)
	
	whiteclip = core.std.BlankClip(src, width=w, height=h, color=[255, 255, 255]).std.ShufflePlanes(0, vs.GRAY)
	edgemask = core.std.Sobel(Yplane)
	edgemask = iterate(edgemask, core.std.Maximum, iter)
	
	return core.std.Expr([whiteclip, edgemask], 'x y -')
	
def descale_rescale(src, w: int, h: int, yuv444=False, descalemode='bicubic', args_a='0.33', args_b='0.33'):
	"""
	src: Input video
	w: Width (int)
	h: Height (int)
	yuv444: True/False (bool)
	descalemode: bicubic, bilinear, spline16. spline36, lanczos(str)
	args_a: Available only for bicubic and lanczos
		- bicubic: b
		- lanczos: taps
	args_b: Available only for bicubic
		- bicubic: c
	Some descale -> upscale -> descale filter.
	Set w & h to native resolution, using debilinear descale for this, enable i444 if you want i444
	"""
	if w is not None and h is not None:
		w = src.width if w is None else w
		h = src.height if h is None else h
	if descalemode.lower() == 'bicubic':
		descale1 = fvf.DebicubicM(src, w, h, b=args_a, c=args_b)
	elif descalemode.lower() == 'bilinear':
		descale1 = fvf.DebilinearM(src, w, h)
	elif descalemode.lower() == 'spline16':
		descale1 = fvf.Despline16M(src, w, h)
	elif descalemode.lower() == 'spline36':
		descale1 = fvf.Despline36M(src, w, h)
	elif descalemode.lower() == 'lanczos':
		if args_a < 2:
			args_a = 3
		descale1 = fvf.DelanczosM(src, w, h, taps=args_a)
	else:
		raise TypeError("descalemode '{}' doesn't exist, use bicubic or bilinear or spline16 or spline36 or lanczos".format(descalemode))
	rpow = edi.nnedi3_rpow2(descale1,rfactor=2) #why not?
	if yuv444:
		if descalemode.lower() == 'bicubic':
			rescale = fvf.DebicubicM(rpow, w, h, b=args_a, c=args_b, yuv444=True)
		elif descalemode.lower() == 'bilinear':
			rescale = fvf.DebilinearM(rpow, w, h, yuv444=True)
		elif descalemode.lower() == 'spline16':
			rescale = fvf.Despline16M(rpow, w, h, yuv444=True)
		elif descalemode.lower() == 'spline36':
			rescale = fvf.Despline36M(rpow, w, h, yuv444=True)
		elif descalemode.lower() == 'lanczos':
			if args_a < 2:
				args_a = 3
			rescale = fvf.DelanczosM(rpow, w, h, taps=args_a, yuv444=True)
	else:
		if descalemode.lower() == 'bicubic':
			rescale = fvf.DebicubicM(rpow, w, h, b=args_a, c=args_b)
		elif descalemode.lower() == 'bilinear':
			rescale = fvf.DebilinearM(rpow, w, h)
		elif descalemode.lower() == 'spline16':
			rescale = fvf.Despline16M(rpow, w, h)
		elif descalemode.lower() == 'spline36':
			rescale = fvf.Despline36M(rpow, w, h)
		elif descalemode.lower() == 'lanczos':
			if args_a < 2:
				args_a = 3
			rescale = fvf.DelanczosM(rpow, w, h, taps=args_a)
	return rescale

def ivtc_deint(clip, field=0, order=1, mode=3, nsize=4, use_gpu=True):
	"""
	Simple IVTC filter
	"""
	def postprocess(n, f, clip, deinterlaced):
	   if f.props['_Combed'] > 0:
	      return deinterlaced
	   else:
	      return clip
	clip = core.vivtc.VFM(clip, order=order, mode=mode)
	if use_gpu:
		clip_deinted = core.nnedi3cl.NNEDI3CL(clip, field=field, nsize=nsize)
	else:
		clip_deinted = core.nnedi3.nnedi3(clip, field=field, nsize=nsize)
	clip = core.std.FrameEval(clip, partial(postprocess, clip=clip, deinterlaced=clip_deinted), prop_src=clip)
	clip = core.vivtc.VDecimate(clip)
	return core.vinverse.Vinverse(clip)

def hdr2sdr(clip,peak=1000,desat=50,brightness=40,cont=1.05,lin=True,show_satmask=False,show_clipped=False):
	import adjust
	"""
	HDR102SDR conversion, using Hable tonemap as it base (Stolen from age@Doom9)
	clip: Video Source
	peak: Video Peak
	desat: Desaturation (I recommend leaving it at 50)
	brightness: Brightness for adjustment (set it to 0 for disable)
	cont: idk what is this, it makes it better tho
	lin: Linearize the tonemap
	satmask: Show Saturation mask (useless)
	clipped: idk (useless)
	"""
	core = vs.get_core()
	c=clip
	c=core.resize.Bicubic(clip=c, format=vs.RGBS,filter_param_a=0.0,filter_param_b=1.0, range_in_s="limited", matrix_in_s="2020ncl", primaries_in_s="2020", primaries_s="2020", transfer_in_s="st2084", transfer_s="linear",dither_type="none", nominal_luminance=peak)
	o=c
	a=c

	source_peak=peak 
	LDR_nits=100     
	exposure_bias=source_peak/LDR_nits

	if (brightness < 0):
		adjustIt = False
	else:
		adjustIt = True
	
	if (brightness > 100):
		brightness = 100

	if (desat < 0) :
		desat=0
	if (desat > 100) :
	   desat=100
	desat=desat/100

	tm=((1*(0.15*1+0.10*0.50)+0.20*0.02) / (1*(0.15*1+0.50)+0.20*0.30)) - 0.02/0.30
	w=((exposure_bias*(0.15*exposure_bias+0.10*0.50)+0.20*0.02)/(exposure_bias*(0.15*exposure_bias+0.50)+0.20*0.30))-0.02/0.30
	tm_ldr_value=tm * (1 / w)#value of 100 nits after the tone mapping
	ldr_value_mult=tm_ldr_value/(1/exposure_bias)#0.1 (100nits) * ldr_value_mult=tm_ldr_value

	tm = core.std.Expr(c, expr="x  {exposure_bias} * 0.15 x  {exposure_bias} * * 0.05 + * 0.004 + x  {exposure_bias} * 0.15 x  {exposure_bias} * * 0.50 + * 0.06 + / 0.02 0.30 / -  ".format(exposure_bias=exposure_bias),format=vs.RGBS)
	w=((exposure_bias*(0.15*exposure_bias+0.10*0.50)+0.20*0.02)/(exposure_bias*(0.15*exposure_bias+0.50)+0.20*0.30))-0.02/0.30
	tm = core.std.Expr(clips=[tm,c], expr="x  1 {w}  / * ".format(exposure_bias=exposure_bias,w=w),format=vs.RGBS)
	tm = core.std.Limiter(tm, 0, 1)

	if lin == True :
		#linearize the tonemapper curve under 100nits
		tm=core.std.Expr(clips=[tm,o], expr="x {tm_ldr_value} < y {ldr_value_mult} * x ? ".format(tm_ldr_value=tm_ldr_value,ldr_value_mult=ldr_value_mult))


	a = core.std.Expr(a, expr="x  {ldr_value_mult} *   ".format(ldr_value_mult=ldr_value_mult),format=vs.RGBS)
	
	r=core.std.ShufflePlanes(clips=[a], planes=[0], colorfamily=vs.GRAY)
	g=core.std.ShufflePlanes(clips=[a], planes=[1], colorfamily=vs.GRAY)
	b=core.std.ShufflePlanes(clips=[a], planes=[2], colorfamily=vs.GRAY)
	#luminance
	l=core.std.Expr(clips=[r,g,b], expr="x 0.2627 *  y 0.678 * + z 0.0593 * +    ",format=vs.GRAY)

	#value under 100nits after the tone mapping(tm_ldr_value) becomes 0, even tm_ldr_value becomes 0 and then scale all in the 0-1 range fo the mask 
	mask=core.std.Expr(clips=[l], expr="x {tm_ldr_value}  - {ldr_value_mult} /".format(ldr_value_mult=ldr_value_mult,tm_ldr_value=tm_ldr_value))
	mask = core.std.Limiter(mask, 0, 1)

	#reduce the saturation blending with grayscale
	lrgb=core.std.ShufflePlanes(clips=[l,l,l], planes=[0,0,0], colorfamily=vs.RGB)
	asat=core.std.Expr(clips=[a,lrgb], expr="y {tm_ldr_value} < x  y {desat} * x 1 {desat} - * +   ?".format(tm_ldr_value=tm_ldr_value,desat=desat))
	a=core.std.MaskedMerge(a, asat, mask)

	r=core.std.ShufflePlanes(clips=[a], planes=[0], colorfamily=vs.GRAY)
	g=core.std.ShufflePlanes(clips=[a], planes=[1], colorfamily=vs.GRAY)
	b=core.std.ShufflePlanes(clips=[a], planes=[2], colorfamily=vs.GRAY)

	rl=core.std.ShufflePlanes(clips=[tm], planes=[0], colorfamily=vs.GRAY)
	gl=core.std.ShufflePlanes(clips=[tm], planes=[1], colorfamily=vs.GRAY)
	bl=core.std.ShufflePlanes(clips=[tm], planes=[2], colorfamily=vs.GRAY)
	l2=core.std.Expr(clips=[rl,gl,bl], expr="x 0.2627 *  y 0.678 * + z 0.0593 * +    ",format=vs.GRAY)
	nl= l2
	scale=core.std.Expr(clips=[nl,l], expr="x y / ")
	r1=core.std.Expr(clips=[r,scale], expr="x  y *")
	g1=core.std.Expr(clips=[g,scale], expr="x  y *")
	b1=core.std.Expr(clips=[b,scale], expr="x  y *")

	c=core.std.ShufflePlanes(clips=[r1,g1,b1], planes=[0,0,0], colorfamily=vs.RGB)
	c = core.std.Limiter(c, 0, 1)

	if show_satmask == True :
		#show mask
		c=core.std.ShufflePlanes(clips=[mask,mask,mask], planes=[0,0,0], colorfamily=vs.RGB)

	if show_clipped == True :
		#show clipped
		c=a
	#then
	c=core.resize.Bicubic(clip=c, format=vs.YUV420P10, filter_param_a=0.0, filter_param_b=1.0, matrix_s="709", primaries_in_s="2020", primaries_s="709", transfer_in_s="linear", transfer_s="709",dither_type="ordered")

	#some adjustment
	if adjustIt:
		if (cont < 0):
			cont = 1.05
		if (cont > 100):
			cont = 100
		c=adjust.Tweak(c,bright=brightness,cont=cont)
	return c 


#aliases
tonemap = hdr2sdr
rescale = descale_rescale


######### Below here is mpeg2stinx script #########

def spline64bob(src, process_chroma=True):
	def bob(src):
		src = src.fmtc.resample(w=src.width, h=src.height, kernel="spline64", css="420").fmtc.bitdepth(bits=8)
		src = src.std.SeparateFields(True)[::2]
		e = core.std.SelectEvery(src, cycle=2, offsets=0).fmtc.resample(w=src.width, h=2*src.height, kernel="spline64", sx=0, sy=0.25, sw=src.width, sh=src.height).fmtc.bitdepth(bits=8)
		o = core.std.SelectEvery(src, cycle=2, offsets=1).fmtc.resample(w=src.width, h=2*src.height, kernel="spline64", sx=0, sy=-0.25, sw=src.width, sh=src.height).fmtc.bitdepth(bits=8)
		return core.std.Interleave(clips=[e,o])
	if src.format == 'YUV420P8':
		return bob(src)
	else:
		y = core.std.ShufflePlanes(clips=src, planes=0, colorfamily=vs.GRAY)
		u = core.std.ShufflePlanes(clips=src, planes=1, colorfamily=vs.GRAY)
		v = core.std.ShufflePlanes(clips=src, planes=2, colorfamily=vs.GRAY)
		if process_chroma:
			return core.std.ShufflePlanes([y,bob(u),bob(v)], planes=[0,0,0], colorfamily=vs.YUV)
		else:
			return core.std.ShufflePlanes([y,core.std.SelectEvery(u,cycle=1, offsets=[0,0]),core.std.SelectEvery(v,cycle=1, offsets=[0,0])], planes=[0,0,0], colorfamily=vs.YUV)

def pointbob(src):
	src = src.std.SeparateFields(True)[::2]
	return core.resize.Point(src, src.width, 2*src.height)

def median3(a,b,c,grey=True):
	return core.std.Interleave([a, b, c]).rgvs.Clense(planes=[0, 1, 2] if grey is True else 0).std.SelectEvery(cycle=3, offsets=1)

def crossfieldrepair(clip, sw=2, sh=2, bobbedClip=None, planes=0, chroma=True):
	if (sw < 0 and sh < 0) or sw < 0 or sh < 0:
		raise ValueError("crossfieldrepair: sw/sh cannot be a negative integers")
	if not isinstance(clip, vs.VideoNode):
		raise vs.Error("crossfieldrepair:\"clip\" not a clip")
	if bobbedClip is None:
		bobbedClip = spline64bob(clip, process_chroma=chroma)
	bob_ex = haf.mt_expand_multi(bobbedClip, planes=1, sw=sw, sh=sh)
	bob_in = haf.mt_inpand_multi(bobbedClip, planes=1, sw=sw, sh=sh)
	if sw == 1 and sh == 1:
		e = core.std.SelectEvery(bobbedClip, cycle=2, offsets=0)
		o = core.std.SelectEvery(bobbedClip, cycle=2, offsets=1)
		re = core.rgvs.Repair(bobbedClip,e,mode=[1])
		ro = core.rgvs.Repair(bobbedClip,o,mode=[1])
	else:
		eX = core.std.SelectEvery(bob_ex, cycle=2, offsets=0)
		oX = core.std.SelectEvery(bob_ex, cycle=2, offsets=1)
		eI = core.std.SelectEvery(bob_in, cycle=2, offsets=0)
		oI = core.std.SelectEvery(bob_in, cycle=2, offsets=1)
		re = median3(bobbedClip,eX,oX, False)
		ro = median3(bobbedClip,eI,oI, False)
	res = core.std.Interleave(clips=[re,ro])
	res = res.std.SeparateFields(True)[::2]
	res = core.std.SelectEvery(res, cycle=4, offsets=[2,1])
	res = res.std.DoubleWeave()[::2]
	return res

def maxyuv(c):
	y = c.resize.Bicubic(width=c.width, height=c.height, format=vs.YUV420P8)
	u = core.resize.Bicubic(core.std.ShufflePlanes(clips=c, planes=1, colorfamily=vs.GRAY), c.width, c.height, vs.YUV420P8)
	v = core.resize.Bicubic(core.std.ShufflePlanes(clips=c, planes=2, colorfamily=vs.GRAY), c.width, c.height, vs.YUV420P8)
	w = c.width
	h = c.height

	yc = core.resize.Bilinear(y, u.width, u.height)
	ls = core.std.Expr([y, core.resize.Bilinear(u,w,h)], expr=['x y max']).std.Expr([y, core.resize.Bilinear(v,w,h)], expr=['x y max'])
	cs = core.std.Expr([yc, u], expr=['x y max']).std.Expr([yc, v], expr=['x y max'])
	return core.std.ShufflePlanes([cs, cs, ls], planes=[0,0,0], colorfamily=vs.YUV)

def mpeg2stinx(src, mode=1, sw=1, sh=1, contra=True, blurv=0.0, sstr=2.0, scl=0.25, dither=False, order=0, diffscl=None):
	#####################################################
	###                                               ###
	###        mpeg2stinx port for VapourSynth        ###
	###                                               ###
	###   		  ported by NoAiOne or N4O            ###
	###   		  originally by torchlight            ###
	###                                               ###
	###                                               ###
	#####################################################
	### This filter is designed to eliminate certain combing-like compression artifacts 
	### that show up all too often in hard-telecined MPEG-2 encodes, 
	### and works to a smaller extent on bitrate-starved hard-telecined AVC encodes as well.
	###
	###
	### +---------+
	### |  USAGE  |
	### +---------+
	###
	### mpeg2stinx(clip, mode, sw, sh, contra, blurv, sstr, scl, dither, order, diffscl)
	###
	### clip: video Source
	### mode: Resizer used for interpolating fields to full size. (0 to 2)
	### sw/sh: Parameters for the size of the rectangle on which to perform min/max clipping
	### contra: Whether to use contrasharpening.
	### blurv: How much vertical blur to apply.
	### sstr: Contrasharpening strength.
	### scl: Contrasharpening scale.
	### dither: Whether to dither when averaging two clips.
	### order: Field order to use for yadifmod.
	### diffscl: If specified, temporal limiting is used, where the changes by crossfieldrepair 
	### 		 are limited to diffscl times the difference between the current frame and its neighbours.

	if not isinstance(src, vs.VideoNode):
		raise ValueError('mpeg2stinx: src is not a video')
	if sw < 0 or sh < 0:
		raise ValueError('mpeg2stinx: sw/sh cannot be a negative integers')
	if mode < 0 or mode > 3:
		raise ValueError('mpeg2stinx: mode must be 0, 1 or 2')
	if order < 0 or order > 2:
		raise ValueError('mpeg2stinx: order must be 0, 1 or 2')
	if diffscl is not None and diffscl >= 0:
		raise ValueError('mpeg2stinx: diffscl must be a negative integers')
	if contra:
		blurv = 1.0
	else:
		blurv = 0.0

	def deint(src, mode, order):
		if mode == 0:
			bobbed = pointbob(src)
		elif mode == 1:
			bobbed = spline64bob(src)
		elif mode == 2:
			bobbed = core.nnedi3.nnedi3(src,field=3)

		if order == 0:
			return bobbed
		elif order == 1:
			return core.std.SelectEvery(core.yadifmod.Yadifmod(src,order=0,mode=3,edeint=core.std.SelectEvery(bobbed, 2, [1,0])).selectevery(2,1,0),2,[1,0])
		elif order == 2:
			return core.yadifmod.Yadifmod(src,order=1,mode=3,edeint=bobbed)

	def templimit(c, flt, ref, diffscl):
		adj = ref.std.SelectEvery(2, [0, 1])
		diff = core.std.Expr(core.std.SelectEvery(c, 3, [0,1]), adj, ["x y - abs"])
		diff = core.std.SeparateFields(True)[::2]
		diff = maxyuv(diff)
		diff2 = core.std.Expr(core.std.SelectEvery(diff, 4, [0,1]), core.std.SelectEvery(diff,4, [2,3]), expr=["x y min"])
		diff2 = haf.mt_expand_multi(diff2,sw=2,sh=1,planes=0)
		diff2 = diff2.std.DoubleWeave()[::2]
		a = core.misc.AverageFrames(clips=[c, diff2], weights=[1, -diffscl])
		b = core.misc.AverageFrames(clips=[c, diff2], weights=[1, diffscl])
		return median3(a,b,flt)
		
	a = crossfieldrepair(src, sw=sw, sh=sh, bobbedClip=deint(src,mode,order))
	if diffscl is not None:
		a = templimit(src,a,src,diffscl)
	b = crossfieldrepair(a,sw=sw,sh=sh,bobbedClip=deint(a,mode,order))
	if diffscl is not None:
		b = templimit(a,b,src,diffscl)

	if dither:
		dit = core.misc.AverageFrames(clips=[a,b], weights=[0.5,0.5])
		dit = mvf.Depth(dit, dither=4)
	else:
		dit = core.misc.AverageFrames(clips=[a, b], weights=[1, 1])

	if blurv > 0.0:
		nuked = core.std.BoxBlur(dit,hradius=1, vradius=blurv)
	else:
		nuked = dit
	
	nukedd = core.std.MakeDiff(src,nuked, [0,1,2])

	sharp = core.std.Expr(nuked,core.std.BoxBlur(nuked,hradius=0,vradius=1).std.BoxBlur(nuked,hradius=0,vradius=0), expr=["x x y - {} * +".format(sstr)])
	sharpd = core.std.Expr(nuked,core.std.BoxBlur(nuked,hradius=0,vradius=1).std.BoxBlur(nuked,hradius=0,vradius=0), expr=["x y - {} * 128 +".format(sstr)])
	limd = core.std.Expr(sharpd,nukedd,expr=[f"x 128 - y 128 - * 0 < {scl} 1 ? x 128 - abs y 128 - abs < x y ? 128 - * 128 +"])

	if scl == 0:
		last = median3(nuked,sharp,src)
	else:
		last = core.std.MergeDiff(nuked,limd,[0,1,2])

	if contra:
		return last
	else:
		return nuked