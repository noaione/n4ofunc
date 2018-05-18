import vapoursynth as vs
import kagefunc as kgf
import fvsfunc as fvf
import mvsfunc as mvf
import vsTAAmbk as taa
import fag3kdb as f3kdb
import havsfunc as hvs
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

#deband
def mSdeband(optIN):
	"""
	optIN: Input video
	Mini-mini version of Deband
	"""
	return f3kdb.Fag3kdb(optIN)
def Sdeband(optIN):
	"""
	optIN: Input video
	Mini version of Deband
	"""
	return f3kdb.Fag3kdb(optIN, radiusy=12, radiusc=8, thry=60, thrc=40, grainy=15, grainc=0)
def Mdeband(optIN):
	"""
	optIN: Input video
	Normal version of Deband
	"""
	return f3kdb.Fag3kdb(optIN, radiusy=18, radiusc=12, thry=60, thrc=50, grainy=15, grainc=0)
def Hdeband(optIN):
	"""
	optIN: Input video
	High version of Deband
	"""
	ref = optIN
	out = optIN.f3kdb.Deband(range=14, y=60, cb=40, cr=40, grainy=15, grainc=0, output_depth=16)
	mask = kgf.retinex_edgemask(ref).std.Binarize(5000).std.Inflate()
	merged = core.std.MaskedMerge(out, ref, mask)
	out = kgf.adaptive_grain(merged,0.20)
	return out

#denoise
def Mknlm(optIN, gpu=0):
	"""
	optIN: Input video
	Uses GPU for encoding
	Normal/Medium version of KNLM
	Better than SMDegrain, worse than BM3D
	"""
	return core.knlm.KNLMeansCL(optIN, a=2, h=0.25, d=3, channels="YUV", device_type='gpu', device_id=0)
def Hknlm(optIN):
	"""
	optIN: Input video
	Uses GPU for Encoding
	High version of KNLM
	Better than SMDegrain, worse than BM3D
	"""
	return core.knlm.KNLMeansCL(optIN, a=2, h=0.4, d=3, s=8, channels="YUV", device_type='gpu', device_id=0)
def Nbm3d(optIN):
	"""
	optIN: Input video
	Normal/Medium version of BM3D
	!!CAUTION!! - Slow af, might break vapoursynth
	REALLY REALLY SLOW, BUT SO FUCKING GOOD
	""" 
	return core.bm3d.Basic(optIN)
def Hbm3d(optIN):
	"""
	optIN: Input video
	High version of BM3D
	!!CAUTION!! - Slow af, might break vapoursynth
	REALLY REALLY SLOW, BUT SO FUCKING GOOD
	"""
	return mvf.BM3D(optIN, sigma=[4,0], radius1=2, matrix=6)
def Nsmd(optIN):
	"""
	optIN: Input video
	Normal/Medium version of SMDegrain
	"""
	return hvs.SMDegrain(optIN, tr=2,thSAD=50,thSADC=75)
def Hsmd(optIN):
	"""
	optIN: Input video
	High version of SMDegrain
	"""
	return hvs.SMDegrain(optIN, tr=2.3, thSAD=50, thSADC=75, prefilter=3, RefineMotion=True, search=4)
def S_hybriddenoise(src, knl=0.4, tr=2, thsad=50, thsadc=75):
	"""
	src: video input
	knl or h: knlmeanscl
	tr, thsad, thsadc: smdegrain
	##Stolen from kageru function
	using smdegrain for denoise luma and knlmeanscl for denoise chroma
	"""
	planes = to_plane_array(src)
	planes[0] = hvs.SMDegrain(planes[0], tr=tr, thSAD=thsad, thSADC=thsadc, prefilter=3, RefineMotion=True, search=4)
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
	planes[0] = mvf.BM3D(planes[0], radius1=radius1, sigma=sigma)
	planes[1], planes[2] = [hvs.SMDegrain(plane, tr=tr, thSAD=thsad, thSADC=thsadc, prefilter=3, RefineMotion=True, search=4)
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

#anti-aliasing
def Nnedi3taa(optIN, cycle=0):
	"""
	optIN: Input video
	cycle = (0-1) - 1 is nightmare
	Fast, and good enough
	"""
	return taa.TAAmbk(optIN,aatype='Nnedi3',cycle=cycle)
def eedi3taa(optIN, cycle=0):
	"""
	optIN: Input video
	cycle = (0-1) - 1 is nightmare
	Slow, but really good, cycle is disabled default
	"""
	return taa.TAAmbk(optIN,aatype='Eedi3',cycle=cycle)
def Ntaa(optIN):
	"""
	optIN: Input video
	Normal kinda one, tho it may broke on some pc
	"""
	return taa.TAAmbk(optIN,aatype=1,mtype=1,strength=0.2,postaa=True,cycle=0)
def Htaa(optIN): #pre-caution, lag.
	"""
	optIN: Input video
	!!!WARNING!!!
	- This one is a fucking nightmare, 1fps encode incoming
	"""
	return taa.TAAmbk(optIN,aatype=1,mtype=1,strength=0.3,postaa=True,cycle=1)

#other
def deblock(optIN):
	"""
	optIN: Input video
	Deblocking stuff for your video
	"""
	return fvf.AutoDeblock(optIN)

#resize
def i444(optIN, w=1280, h=720):
	"""
	optIN: Input Video
	w: Width Resolution
	h: Height Resoulution
	Hi444PP meme resize filter, using spline36 as kernely and blackmanminlobe for kerneluv
	"""
	return fvf.Downscale444(optIN, w, h, kernely='blackmanminlobe', kerneluv='blackmanminlobe')
def descale_rescale(src, w: int, h: int, yuv444=False):
	"""
	optIN: Input video
	w: Width
	h: Height
	Some descale -> upscale -> descale filter.
	Set w & h to native resolution, using debilinear descale for this, enable i444 if you want i444
	"""
	if w is not None and h is not None:
		w = src.ViedoNode.width if w is None else w
		h = src.VideoNode.height if h is None else h

		descale1 = fvf.DebicubicM(src, w, h, b=0, c=1)
		rpow = edi.nnedi3_rpow2(descale1,rfactor=2) #why not?
		if yuv444:
			rescale = i444(rpow,w,h)
		else:
			rescale = fvf.Despline36M(rpow, w, h)
	return rescale

#etc
def dehardsub(softVideo, hardVideo):
	"""
	softVideo: Soft-subbed video
	hardVideo: Hard-subbed video
	- tl;dr Dehardsubbing filter
	"""
	mask = kgf.hardsubmask(hardVideo,softVideo)
	mask = fvf.Depth(mask, 16)
	return core.std.MaskedMerge(hardVideo,softVideo,mask)

def animu(optIN, field=1, order=1, mode=3):
	"""
	Anime IVTC filterino
	optIN: Input Video
	order: 0 is bottom field first, 1 is top field first
	field: 1
	mode: see this, http://www.vapoursynth.com/doc/plugins/vivtc.html
	"""
	ivtcanimu = [core.vivtc.VFM(optIN, order=order, mode=mode)]
	dein = [core.nnedi3.nnedi3(ivtcanimu, field=field)]
	animu = [core.vivtc.VDecimate(dein)]
	return core.resize.Spline36(animu, 1280, 720, format=vs.YUV420P10)

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

def benchmark(optIN, w=1280, h=720):
	"""
	Sole PC test purpose lol
	optIN is Video
	w: width
	h: height
	default is 1280x720
	"""
	src = fvf.Depth(optIN, 16)
	rpow = edi.nnedi3_rpow2(src,rfactor=2) #why not?
	denoise = M_hybriddenoise(rpow)
	deband = f3kdb.Fag3kdb(denoise, radiusy=12, radiusc=8, thry=60, thrc=40, grainy=15, grainc=0)
	i444 = fvf.Downscale444(deband, w=w, h=h, kernely="blackmanminlobe", kerneluv="blackmanminlobe")
	aa = taa.TAAmbk(i444,aatype='Eedi3',cycle=1)
	finalmeme = fvf.Depth(aa, 10)
	return finalmeme.set_output()

#aliases
tonemap = hdr2sdr
autistCode = benchmark
ivtc = animu

"""
(c) 2018 Autist version of N4O
"""

