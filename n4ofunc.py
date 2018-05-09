import vapoursynth as vs
import kagefunc as kgf
import fvsfunc as fvf
import mvsfunc as mvf
import vsTAAmbk as taa
import fag3kdb as f3kdb
import havsfunc as hvs
import edi_rpow2 as edi
import functools

#init
core = vs.core

#deband
def mSdeband(optIN : vs.VideoNode):
	"""
	optIN: Input video
	Mini-mini version of Deband
	"""
	debandsimpleaf = f3kdb.Fag3kdb(optIN)
	return debandsimpleaf
def Sdeband(optIN : vs.VideoNode):
	"""
	optIN: Input video
	Mini version of Deband
	"""
	debandsimple = f3kdb.Fag3kdb(optIN, radiusy=12, radiusc=8, thry=60, thrc=40, grainy=15, grainc=0)
	return debandsimple
def Mdeband(optIN : vs.VideoNode):
	"""
	optIN: Input video
	Normal version of Deband
	"""
	debandsimple = f3kdb.Fag3kdb(optIN, radiusy=18, radiusc=12, thry=60, thrc=50, grainy=15, grainc=0)
	return debandsimple
def Hdeband(optIN : vs.VideoNode):
	"""
	optIN: Input video
	High version of Deband
	"""
	ref = optIN
	out = optIN.f3kdb.Deband(range=16, y=50, cb=25, cr=25, grainy=15, grainc=0, output_depth=16)
	mask = kgf.retinex_edgemask(ref).std.Inflate()
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
		return core.knlm.KNLMeansCL(optIN, a=2, h=0.25, d=3, channels="YUV", device_type='gpu', device_id=0)
	elif gpu == 0:
		return core.knlm.KNLMeansCL(optIN, a=2, h=0.25, d=3, channels="YUV")
	else:
		raise ValueError("Number defined is out of range")
def Hknlm(optIN : vs.VideoNode, gpu=0):
	"""
	optIN: Input video
	gpu = 0 for disable using gpu, 1 for enable using gpu
	High version of KNLM
	Better than SMDegrain, worse than BM3D
	"""
	if gpu == 1:
		return core.knlm.KNLMeansCL(optIN, a=2, h=0.4, d=3, s=8, channels="YUV", device_type='gpu', device_id=0)
	elif gpu == 0:
		return core.knlm.KNLMeansCL(optIN, a=2, h=0.4, d=3, s=8, channels="YUV")
	else:
		raise ValueError("Number defined is out of range")
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
	hbm = mvf.BM3D(optIN, sigma=[4,0], radius1=2, matrix=6)
	return hbm
def Nsmd(optIN : vs.VideoNode):
	"""
	optIN: Input video
	Normal/Medium version of SMDegrain
	"""
	smdN = hvs.SMDegrain(optIN, tr=2,thSAD=50,thSADC=75)
	return smdN
def Hsmd(optIN : vs.VideoNode):
	"""
	optIN: Input video
	High version of SMDegrain
	"""
	smh = hvs.SMDegrain(optIN, tr=0.3, prefilter=3, RefineMotion=True, search=4)
	return smh
def Nnedi3taa(optIN : vs.VideoNode, cycle=0):
	"""
	optIN: Input video
	cycle = (0-1) - 1 is nightmare
	Fast, and good enough
	"""
	taaN = taa.TAAmbk(optIN,aatype='Nnedi3',cycle=cycle)
	return taaN
def eedi3taa(optIN : vs.VideoNode, cycle=0):
	"""
	optIN: Input video
	cycle = (0-1) - 1 is nightmare
	Slow, but really good, cycle is disabled default
	"""
	taaE = taa.TAAmbk(optIN,aatype='Eedi3',cycle=cycle)
	return taaE
def Ntaa(optIN : vs.VideoNode):
	"""
	optIN: Input video
	Normal kinda one, tho it may broke on some pc
	"""
	taaNor = taa.TAAmbk(optIN,aatype=1,mtype=1,strength=0.2,postaa=True,cycle=0)
	return taaNor
def Htaa(optIN : vs.VideoNode): #caution, autist level bypassing through the fucking roof
	"""
	optIN: Input video
	!!!WARNING!!!
	- This one is a fucking nightmare, 1fps encode incoming
	"""
	taaHi = taa.TAAmbk(optIN,aatype=1,mtype=1,strength=0.3,postaa=True,cycle=1)
	return taaHi
def deblock(optIN : vs.VideoNode):
	"""
	optIN: Input video
	Deblocking stuff for your video
	"""
	debl = fvf.AutoDeblock(optIN)
	return debl
#resize
def i444(optIN : vs.VideoNode, wRes=1280, hRes=720):
	"""
	optIN: Input Video
	wRes: Width Resolution
	hRes: Height Resoulution
	Hi444PP meme resize filter, using spline36 as kernely and blackmanminlobe for kerneluv
	"""
	i4444 = fvf.Downscale444(optIN, wRes, hRes, kernely='Spline36', kerneluv='blackmanminlobe')
	return i4444
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
def ivtcresize(optIN: vs.VideoNode):
	return core.resize.Spline36(optIN, 1280, 720, format=vs.YUV420P10)
def ivtc(optIN: vs.VideoNode, order=1, mode=3):
	return core.vivtc.VFM(optIN, order=order, mode=mode)
def deinterlace(optIN : vs.VideoNode,field=1):
	return core.nnedi3.nnedi3(optIN,field=field)
def animu(optIN: vs.VideoNode):
	"""
	Anime IVTC filterino
	optIN: Input Video
	order: 0 is bottom field first, 1 is top field first
	mode: see this, http://www.vapoursynth.com/doc/plugins/vivtc.html
	"""
	ivtcanimu = ivtc(optIN)
	dein = deinterlace(ivtcanimu)
	animu = ivtcresize(dein)
	return core.vivtc.VDecimate(animu)
def tonemap1(optIN: vs.VideoNode):
	dither32 = fvf.Depth(optIN, 32)
	return core.tonemap.Mobius(dither32, exposure=1.35, transition=10, peak=25)
def hdr2sdr(optIN: vs.VideoNode):
	"""
	HDR2SDR Conversion
	optIN: Video input
	!!!WARNING!!!
	VERY EXPERIMENTAL THINGY RIGHT HERE
	"""
	dither32 = fvf.Depth(optIN, 32) #dither to 32bit because tonemap only supported this bitdepth
	mobius = tonemap1(dither32)
	return core.tonemap.Reinhard(mobius, exposure=2.0, contrast=0.45, peak=1.7)
def autistCode(optIN : vs.VideoNode, gpu=0):
	"""
	Sole PC test purpose lol
	optIN is Video
	gpu: 1 is enable, 0 is disable
	Just add yourvar.set_output() after using this script
	"""
	src = fvf.Depth(optIN, 16)
	rpow = edi.nnedi3_rpow2(src,rfactor=2) #why not?
	if gpu == 1:
		knlmM = core.knlm.KNLMeansCL(rpow, a=2, h=0.4, d=3, s=8, channels="YUV", device_type='gpu', device_id=0)
	elif gpu == 0:
		knlmM = core.knlm.KNLMeansCL(rpow, a=2, h=0.4, d=3, s=8, channels="YUV")
	else:
		raise ValueError('GPU number is out of range')
	deband = f3kdb.Fag3kdb(knlmM, radiusy=12, radiusc=8, thry=60, thrc=40, grainy=15, grainc=0)
	i444 = fvf.Downscale444(deband, w=1280, h=720, kernely="blackmanminlobe", kerneluv="blackmanminlobe")
	aa = taa.TAAmbk(i444,aatype=1,mtype=1)
	finalmeme = fvf.Depth(aa, 10)
	return finalmeme

"""
(c) 2018 Autist version of N4O
"""