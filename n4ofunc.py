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
def mSdeband(optIN : vs.VideoNode):
	"""
	optIN: Input video
	Mini-mini version of Deband
	"""
	return f3kdb.Fag3kdb(optIN)
def Sdeband(optIN : vs.VideoNode):
	"""
	optIN: Input video
	Mini version of Deband
	"""
	return f3kdb.Fag3kdb(optIN, radiusy=12, radiusc=8, thry=60, thrc=40, grainy=15, grainc=0)
def Mdeband(optIN : vs.VideoNode):
	"""
	optIN: Input video
	Normal version of Deband
	"""
	return f3kdb.Fag3kdb(optIN, radiusy=18, radiusc=12, thry=60, thrc=50, grainy=15, grainc=0)
def Hdeband(optIN : vs.VideoNode):
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
def Mknlm(optIN : vs.VideoNode, gpu=0):
	"""
	optIN: Input video
	gpu = 0 for disable using gpu, 1 for enable using gpu
	Normal/Medium version of KNLM
	Better than SMDegrain, worse than BM3D
	"""
	if gpu == 1:
		denoise = [core.knlm.KNLMeansCL(optIN, a=2, h=0.25, d=3, channels="YUV", device_type='gpu', device_id=0)]
	elif gpu == 0:
		denoise = [core.knlm.KNLMeansCL(optIN, a=2, h=0.25, d=3, channels="YUV")]
	else:
		raise ValueError("Number defined is out of range")
	return denoise
def Hknlm(optIN : vs.VideoNode, gpu=0):
	"""
	optIN: Input video
	gpu = 0 for disable using gpu, 1 for enable using gpu
	High version of KNLM
	Better than SMDegrain, worse than BM3D
	"""
	if gpu == 1:
		denoise = [core.knlm.KNLMeansCL(optIN, a=2, h=0.4, d=3, s=8, channels="YUV", device_type='gpu', device_id=0)]
	elif gpu == 0:
		denoise = [core.knlm.KNLMeansCL(optIN, a=2, h=0.4, d=3, s=8, channels="YUV")]
	else:
		raise ValueError("Number defined is out of range")
	return denoise
def Nbm3d(optIN : vs.VideoNode):
	"""
	optIN: Input video
	Normal/Medium version of BM3D
	!!CAUTION!! - Slow af, might break vapoursynth
	REALLY REALLY SLOW, BUT SO FUCKING GOOD
	""" 
	return core.bm3d.Basic(optIN)
def Hbm3d(optIN : vs.VideoNode):
	"""
	optIN: Input video
	High version of BM3D
	!!CAUTION!! - Slow af, might break vapoursynth
	REALLY REALLY SLOW, BUT SO FUCKING GOOD
	"""
	return mvf.BM3D(optIN, sigma=[4,0], radius1=2, matrix=6)
def Nsmd(optIN : vs.VideoNode):
	"""
	optIN: Input video
	Normal/Medium version of SMDegrain
	"""
	return hvs.SMDegrain(optIN, tr=2,thSAD=50,thSADC=75)
def Hsmd(optIN : vs.VideoNode):
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
def Nnedi3taa(optIN : vs.VideoNode, cycle=0):
	"""
	optIN: Input video
	cycle = (0-1) - 1 is nightmare
	Fast, and good enough
	"""
	return taa.TAAmbk(optIN,aatype='Nnedi3',cycle=cycle)
def eedi3taa(optIN : vs.VideoNode, cycle=0):
	"""
	optIN: Input video
	cycle = (0-1) - 1 is nightmare
	Slow, but really good, cycle is disabled default
	"""
	return taa.TAAmbk(optIN,aatype='Eedi3',cycle=cycle)
def Ntaa(optIN : vs.VideoNode):
	"""
	optIN: Input video
	Normal kinda one, tho it may broke on some pc
	"""
	return taa.TAAmbk(optIN,aatype=1,mtype=1,strength=0.2,postaa=True,cycle=0)
def Htaa(optIN : vs.VideoNode): #caution, autist level bypassing through the fucking roof
	"""
	optIN: Input video
	!!!WARNING!!!
	- This one is a fucking nightmare, 1fps encode incoming
	"""
	return taa.TAAmbk(optIN,aatype=1,mtype=1,strength=0.3,postaa=True,cycle=1)

#other
def deblock(optIN : vs.VideoNode):
	"""
	optIN: Input video
	Deblocking stuff for your video
	"""
	return fvf.AutoDeblock(optIN)

#resize
def i444(optIN : vs.VideoNode, w=1280, h=720):
	"""
	optIN: Input Video
	w: Width Resolution
	h: Height Resoulution
	Hi444PP meme resize filter, using spline36 as kernely and blackmanminlobe for kerneluv
	"""
	return fvf.Downscale444(optIN, w, h, kernely='blackmanminlobe', kerneluv='blackmanminlobe')
def descale_rescale(src : vs.VideoNode, w: int, h: int, yuv444=False):
	"""
	optIN: Input video
	w: Width
	h: Height
	Some descale -> upscale -> descale filter.
	Set w & h to native resolution, using debilinear descale for this, enable i444 if you want i444
	"""
	descale1 = fvf.DebicubicM(src, w, h, b=0, c=1)
	rpow = edi.nnedi3_rpow2(descale1,rfactor=2) #why not?
	if yuv444:
		rescale = i444(rpow,w,h)
	else:
		rescale = fvf.Despline36M(rpow, w, h)
	return rescale

#etc
def dehardsub(softVideo : vs.VideoNode, hardVideo : vs.VideoNode):
	"""
	softVideo: Soft-subbed video
	hardVideo: Hard-subbed video
	- tl;dr Dehardsubbing filter
	"""
	mask = kgf.hardsubmask(hardVideo,softVideo)
	mask = fvf.Depth(mask, 16)
	return core.std.MaskedMerge(hardVideo,softVideo,mask)

def animu(optIN: vs.VideoNode, field=1, order=1, mode=3):
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

def hdr2sdr(optIN: vs.VideoNode):
	"""
	HDR2SDR Conversion
	optIN: Video input
	!!!WARNING!!!
	VERY EXPERIMENTAL THINGY RIGHT HERE
	"""
	dither32 = fvf.Depth(optIN, 32) #dither to 32bit because tonemap only supported this bitdepth
	mobius = [core.tonemap.Mobius(dither32, exposure=1.35, transition=10, peak=25)]
	return core.tonemap.Reinhard(mobius, exposure=2.0, contrast=0.45, peak=1.7)

def benchmark(optIN : vs.VideoNode, w=1280, h=720):
	"""
	Sole PC test purpose lol
	optIN is Video
	w: width
	h: height
	default is 1280x720
	Just add yourvar.set_output() after using this script
	"""
	src = fvf.Depth(optIN, 16)
	rpow = edi.nnedi3_rpow2(src,rfactor=2) #why not?
	denoise = M_hybriddenoise(rpow)
	deband = f3kdb.Fag3kdb(denoise, radiusy=12, radiusc=8, thry=60, thrc=40, grainy=15, grainc=0)
	i444 = fvf.Downscale444(deband, w=w, h=h, kernely="blackmanminlobe", kerneluv="blackmanminlobe")
	aa = taa.TAAmbk(i444,aatype='Eedi3',cycle=1)
	finalmeme = fvf.Depth(aa, 10)
	return finalmeme

"""
(c) 2018 Autist version of N4O
"""