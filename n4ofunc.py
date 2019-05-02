from functools import partial

import nnedi3_rpow2 as edi
import fvsfunc as fvf
import havsfunc as haf
import kagefunc as kgf
import mvsfunc as mvf
import vapoursynth as vs

from MaskDetail import maskDetail
from vapoursynth import core
from vsutil import get_y, is_image, iterate, split, get_w
from math import ceil, log

#helper
def is_extension(x, y):
    """
    Return a boolean if extension are the same or not
    `x` are lowered/downcased
    """
    return x.lower()[x.lower().rfind('.'):] == y


def register_f(c: vs.VideoNode, yuv444=False) -> vs.VideoNode.format:
    """
    Return a registered new format
    """
    return core.register_format(
        c.format.color_family,
        c.format.sample_type,
        c.format.bits_per_sample,
        0 if yuv444 else c.format.subsampling_w,
        0 if yuv444 else c.format.subsampling_w
    )


def nDeHalo(clp=None, rx=None, ry=None, darkstr=None, brightstr=None, lowsens=None, highsens=None, ss=None, thr=None):
    """
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
    """
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


def save_difference(src1, src2, threshold=0.1):
    """
    n4ofunc.save_difference

    Save a difference between src1 and src2
    Useful for comparing between TV and BD

    :src1 vapoursynth.VideoNode: Video Source 1 as the "old" video
    :src2 vapoursynth.VideoNode: Video Source 2 as the "new" video
    :threshold float: Luma threshold between src1 and src2
    """
    import cv2
    import numpy as np
    import os
    import shutil

    src1_cf = src1.format.color_family
    src2_cf = src2.format.color_family
    src1_bits = src1.format.bits_per_sample
    src2_bits = src2.format.bits_per_sample

    if src1.num_frames != src2.num_frames:
        raise ValueError('src1 and src2 frame are not the same')

    def save_as_image(vid, frame, output_format):
        vid = vid.get_frame(frame)
        planes = vid.format.num_planes
        v = cv2.merge([np.array(vid.get_read_array(i), copy=False) for i in reversed(range(planes))])
        v = cv2.resize(v, (1280, 720))
        cv2.imwrite(output_format.format(frame), v)

    cwd = os.getcwd()
    dirsave = cwd + '\\frame_difference'
    print('@@ save_difference: starting process')

    if not os.path.isdir(dirsave):
        os.mkdir(dirsave)

    if src1_cf != vs.RGB:
        src1 = mvf.ToRGB(src1)
    if src2_cf != vs.RGB:
        src2 = mvf.ToRGB(src2)

    print('Converted to RGB')

    if src1_bits != 8:
        src1 = fvf.Depth(src1, 8)
    if src2_bits != 8:
        src2 = fvf.Depth(src2, 8)

    src1_gray = src1.std.ShufflePlanes(0, vs.GRAY)
    src2_gray = src2.std.ShufflePlanes(0, vs.GRAY)
    print('grayed')

    n = 0
    for i, f in enumerate(core.std.PlaneStats(src1_gray, src2_gray).frames()):
        print('@@ save_difference: Processing Frame {}/{} ({})'.format(i, src1_gray.num_frames, f.props["PlaneStatsDiff"]), end='\r')
        if f.props["PlaneStatsDiff"] >= threshold:
            print('@@ save_difference: Diff frame: {}'.format(i), end='\r\n')
            save_as_image(src1, i, dirsave + '\\{}_src1.png')
            save_as_image(src2, i, dirsave + '\\{}_src2.png')
            n += 1

    if n == 0:
        print('@@ save_difference: no significant difference found from current threshold')
        shutil.rmtree(dirsave)

    return


def adaptive_smdegrain(src, thSAD=None, thSADC=None, luma_scaling=None, area='light', iter_edge=0, show_mask=False):
    """
    Some adaptive degrain that will degrain according to area determined (will not affect lineart)
    """
    if thSAD is None:
        thSAD = 150
    if thSADC is None:
        thSADC = thSAD

    if luma_scaling is None:
        luma_scaling = 30

    if area not in ['dark', 'light']:
        raise ValueError('n4ofunc.adaptive_smdegrain: `area` can only be: `light` and `dark`')

    adaptmask = kgf.adaptive_grain(src, luma_scaling=luma_scaling, show_mask=True)

    y_plane = get_y(src)

    if area == 'light':
        adaptmask = adaptmask.std.Invert()

    limitx = y_plane.std.Convolution([-1, -2, -1, 0, 0, 0, 1, 2, 1], saturate=False)
    limity = y_plane.std.Convolution([-1, 0, 1, -2, 0, 2, -1, 0, 1], saturate=False)
    limit = core.std.Expr([limitx, limity], 'x y max')
    limit = iterate(limit, core.std.Maximum, iter_edge)

    mask = core.std.Expr([adaptmask, limit], 'x y -')

    if show_mask:
        return mask

    fil = haf.SMDegrain(src, thSAD=thSAD, thSADC=thSADC)

    return core.std.MaskedMerge(src, fil, mask)


def antiedgemask(src, iteration=1):
    """
    Make some anti-edge mask that use whiteclip and Sobel mask and minus them together
    """
    w = src.width
    h = src.height

    y_plane = get_y(src)

    whiteclip = core.std.BlankClip(src, width=w, height=h, color=[255, 255, 255]).std.ShufflePlanes(0, vs.GRAY)
    edgemask = core.std.Sobel(y_plane)
    edgemask = iterate(edgemask, core.std.Maximum, iteration)

    return core.std.Expr([whiteclip, edgemask], 'x y -')


def masked_descale(src, target_w=None, target_h=None, kernel='bicubic', b=1/3, c=1/3, taps=3, yuv444=False, expandN=2, inflateN=1, iter_max=1, masked=True, show_mask=False) -> vs.VideoNode:
    if not src:
        raise ValueError('src cannot be empty')
    if not target_w:
        raise ValueError('target_w cannot be empty')
    if not target_h:
        raise ValueError('target_h cannot be empty')

    iter_max -= 1

    if iter_max < 0:
        raise ValueError('iter_max cannot be negative integer')

    kernel = kernel.lower()

    if kernel not in ['bicubic', 'bilinear', 'lanczos', 'spline16', 'spline36']:
        raise ValueError('Kernel type doesn\'t exist\nAvailable one: bicubic, bilinear, lanczos, spline16, spline36')

    def VideoResizer(b, c, taps, kernel):
        if kernel == 'bilinear':
            return core.resize.Bilinear
        elif kernel == 'bicubic':
            return partial(core.resize.Bicubic, filter_param_a=b, filter_param_b=c)
        elif kernel == 'lanczos':
            return partial(core.resize.Lanczos, filter_param_a=taps)
        elif kernel == 'spline16':
            return core.resize.Spline16
        elif kernel == 'spline36':
            return core.resize.Spline36

    def DescaleVideo(src, width, height, kernel='bilinear', b=1/3, c=1/3, taps=3, yuv444=False, gray=False, chromaloc=None):
        #####################################
        ##                                 ##
        ##  Source code from `descale.py`  ##
        ##                                 ##
        #####################################
        def get_filter(b, c, taps, kernel):
            if kernel.lower() == 'bilinear':
                return core.descale.Debilinear
            elif kernel.lower() == 'bicubic':
                return partial(core.descale.Debicubic, b=b, c=c)
            elif kernel.lower() == 'lanczos':
                return partial(core.descale.Delanczos, taps=taps)
            elif kernel.lower() == 'spline16':
                return core.descale.Despline16
            elif kernel.lower() == 'spline36':
                return core.descale.Despline36
        src_f = src.format
        src_cf = src_f.color_family
        src_st = src_f.sample_type
        src_bits = src_f.bits_per_sample
        src_sw = src_f.subsampling_w
        src_sh = src_f.subsampling_h

        descale_filter = get_filter(b, c, taps, kernel)

        if src_cf == vs.RGB and not gray:
            rgb = descale_filter(src.resize.Point(format=vs.RGBS), width, height)
            return rgb.resize.Point(format=src_f.id)

        y = descale_filter(src.resize.Point(format=vs.GRAYS), width, height)
        y_f = core.register_format(vs.GRAY, src_st, src_bits, 0, 0)
        y = y.resize.Point(format=y_f.id)

        if src_cf == vs.GRAY or gray:
            return y

        if not yuv444 and ((width % 2 and src_sw) or (height % 2 and src_sh)):
            raise ValueError('Descale: The output dimension and the subsampling are incompatible.')

        uv_f = register_f(src, yuv444)
        uv = src.resize.Spline36(width, height, format=uv_f.id, chromaloc_s=chromaloc)

        return core.std.ShufflePlanes([y,uv], [0,1,2], vs.YUV)

    descale = DescaleVideo(src, target_w, target_h, kernel, b, c, taps, yuv444)
    descale_format = register_f(descale, yuv444)
    VidResize = VideoResizer(b, c, taps, kernel)
    nr_resize = VidResize(src, target_w, target_h, format=descale_format.id)

    video_mask = maskDetail(src, target_w, target_h, expandN=expandN, inflateN=inflateN, kernel=kernel)
    video_mask = iterate(video_mask, core.std.Maximum, iter_max)

    if show_mask:
        return video_mask

    if masked:
        return core.std.MaskedMerge(descale, nr_resize, video_mask)
    return descale


def source(src, lsmas=False, depth=False, trims=None, dither_yuv=True) -> vs.VideoNode:
    """
    Open Video or Image Source
    :param src: Video or Image source (format: string)
    :param lsmas: force use lsmas (file with .m2ts extension will forced to use lsmas)
    :param depth: Dither video (Disable with False, default: False (Use original bitdepth))
    :param trims: Trim video (Integer + List type)
    :param dither_yuv: Dither Image or Video to YUV subsample
    """
    def parse_trim_data(trim_data, video):
        a, b = trim_data

        if a == 'black24':
            a = 24
        if b == 'black24':
            b = video.num_frames - 25

        if a == 'black12':
            a = 12
        if b == 'black12':
            b = video.num_frames - 13

        # Every retarded streaming sites ever
        if a == 'funi':
            a = 240
        if b == 'sentai':
            b = video.num_frames - 1458 # May be changed idk

        if b == 'last':
            b = video.num_frames - 1

        if b < 0:
            b = video.num_frames - (abs(b) + 1)

        return a, b


    def Source(src, lsmas=False):
        if is_extension(src, '.m2ts') or is_extension(src, '.ts'): # Force lsmas
            lsmas = True
        if is_extension(src, '.d2v'):
            return core.d2v.Source
        if is_extension(src, '.avi'):
            return core.avisource.AVISource
        if is_image(src):
            return core.imwri.Read
        if lsmas:
            return core.lsmas.LWLibavSource
        return core.ffms2.Source


    if not isinstance(lsmas, bool):
        return ValueError('lsmas: boolean only (True or False)')
    if not isinstance(src, str):
        return ValueError('src: must be string input')

    OpenSource = Source(src, lsmas)
    src = OpenSource(src)

    if dither_yuv and src.format.color_family != vs.YUV:
        src = core.resize.Point(src, format=vs.YUV420P8, matrix_s='709')

    if depth:
        src = fvf.Depth(src, depth)
    if trims:
        first, last = parse_trim_data(trims, src)
        return core.std.Trim(src, first, last)
    return src


def adaptive_scaling(clip: vs.VideoNode, target_w=None, target_h=None, descale_range=[], kernel='bicubic', b=1/3, c=1/3, taps=3, iter_max=3, rescale=True, show_native_res=False, show_mask=False):
    """
    n4ofunc.adaptive_scaling
    Descale within range and upscale it back to target_w and target_h
    If target are not defined, it will be using original resolution

    Written originally by kageru, modified by NoAiOne.

    :vapoursynth.VideoNode clip:
    :int target_w: Target upscaled width resolution
    :int target_h: Target upscaled width resolution
    :list descale_range: Descale range number in list
    :str kernel: Descaling kernel
    :float b: Bicubic Descale "b" num
    :float c: Bicubic Descale "c" num
    :int taps: Lanczos Descale "c" num
    :int iter_max: Iterate mask with core.std.Maximum()
    :bool show_mask: Show mask
    """
    target_w = clip.width if target_w is None else target_w
    target_h = clip.height if target_h is None else target_h
    if not isinstance(descale_range, list):
        raise TypeError('adaptive_scaling: descale_range: must be a list containing 2 number')

    if len(descale_range) < 2:
        raise ValueError('adaptive_scaling: descale_range: need 2 different number to start')

    if descale_range[0] > descale_range[1]:
        raise ValueError('adaptive_scaling: descale_range: first value cannot be larger than second value')

    if descale_range[0] > target_h and rescale or descale_range[1] > target_h and rescale:
        raise ValueError('adaptive_scaling: descale_range: One of the value cannot be larger than target_h')

    if kernel not in ['bicubic', 'bilinear', 'lanczos', 'spline16', 'spline36']:
        raise ValueError('adaptive_scaling: kernel: Kernel type doesn\'t exist\nAvailable one: bicubic, bilinear, lanczos, spline16, spline36')

    if (target_w % 2) != 0:
        raise ValueError('adaptive_scaling: target_w: Must be a mod2 number (even number)')

    if (target_h % 2) != 0:
        raise ValueError('adaptive_scaling: target_h: Must be a mod2 number (even number)')

    def VideoResizer(b, c, taps, kernel):
        if kernel == 'bilinear':
            return core.resize.Bilinear
        elif kernel == 'bicubic':
            return partial(core.resize.Bicubic, filter_param_a=b, filter_param_b=c)
        elif kernel == 'lanczos':
            return partial(core.resize.Lanczos, filter_param_a=taps)
        elif kernel == 'spline16':
            return core.resize.Spline16
        elif kernel == 'spline36':
            return core.resize.Spline36

    def VideoDescaler(b, c, taps, kernel):
        if kernel.lower() == 'bilinear':
            return core.descale.Debilinear
        elif kernel.lower() == 'bicubic':
            return partial(core.descale.Debicubic, b=b, c=c)
        elif kernel.lower() == 'lanczos':
            return partial(core.descale.Delanczos, taps=taps)
        elif kernel.lower() == 'spline16':
            return core.descale.Despline16
        elif kernel.lower() == 'spline36':
            return core.descale.Despline36

    def simple_descale(y: vs.VideoNode, h: int) -> tuple:
        down = global_clip_descaler(y, get_w(h), h)
        up = global_clip_resizer(down, target_w, target_h)
        diff = core.std.Expr([y, up], 'x y - abs').std.PlaneStats()
        return down, diff

    ref = clip
    clip32 = fvf.Depth(clip, 32)
    y = get_y(clip32)
    global_clip_resizer = VideoResizer(b, c, taps, kernel)
    global_clip_descaler = VideoDescaler(b, c, taps, kernel)

    descale_listp = [simple_descale(y, h) for h in range(descale_range[0], descale_range[1])]
    descale_list = [a[0] for a in descale_listp]
    descale_props = [a[1] for a in descale_listp]

    def select(n, descale_list, f):
        errors = [x.props.PlaneStatsAverage for x in f]
        y_deb = descale_list[errors.index(min(errors))]
        dmask = core.std.Expr([y, global_clip_resizer(y_deb, target_w, target_h)], 'x y - abs 0.025 > 1 0 ?').std.Maximum()
        y_deb16 = fvf.Depth(y_deb, 16)

        y_scaled = edi.nnedi3_rpow2(y_deb16, nns=4, correct_shift=True, width=target_w, height=target_h).fmtc.bitdepth(bits=32)
        if show_native_res and not show_mask:
            y_scaled = core.text.Text(y_scaled, 'Native resolution for this frame: {}'.format(y_deb.height))
        return core.std.ClipToProp(y_scaled, dmask)

    y_deb = core.std.FrameEval(y, partial(select, descale_list=descale_list), prop_src=descale_props)
    dmask = core.std.PropToClip(y_deb)

    def square():
        top = core.std.BlankClip(length=len(y), format=vs.GRAYS, height=4, width=10, color=[1])
        side = core.std.BlankClip(length=len(y), format=vs.GRAYS, height=2, width=4, color=[1])
        center = core.std.BlankClip(length=len(y), format=vs.GRAYS, height=2, width=2, color=[0])
        t1 = core.std.StackHorizontal([side, center, side])
        return core.std.StackVertical([top, t1, top])

    line = core.std.StackHorizontal([square()] * (target_w // 10))
    full = core.std.StackVertical([line] * (target_h // 10))

    artifacts = core.misc.Hysteresis(global_clip_resizer(dmask, target_w, target_h, _format=vs.GRAYS), core.std.Expr([get_y(clip32).tcanny.TCanny(sigma=3), full], 'x y min'))

    ret_raw = kgf.retinex_edgemask(ref)
    ret = ret_raw.std.Binarize(30).rgvs.RemoveGrain(3)
    mask = core.std.Expr([iterate(artifacts, core.std.Maximum, iter_max), ret.resize.Point(_format=vs.GRAYS)], 'y x -').std.Binarize(0.4)
    mask = mask.std.Inflate().std.Convolution(matrix=[1] * 9).std.Convolution(matrix=[1] * 9)

    if show_mask:
        return mask

    merged = core.std.MaskedMerge(y, y_deb, mask)
    merged = core.std.ShufflePlanes([merged, clip32], [0, 1, 2], vs.YUV)
    return fvf.Depth(merged, 16)


src = source
descale = masked_descale
antiedge = antiedgemask
adaptive_degrain = adaptive_smdegrain
adaptive_rescale = partial(adaptive_scaling, rescale=True)
#adaptive_descale = partial(adaptive_scaling, rescale=False) # will be written later


######### Below here is mpeg2stinx script #########

# Note: Not working currently, help me

def spline64bob(src, process_chroma=True):
    def bob(src):
        src = src.fmtc.resample(w=src.width, h=src.height, kernel="spline64", css="420").fmtc.bitdepth(bits=8)
        src = src.std.SeparateFields(True)[::2]
        e = core.std.SelectEvery(src, cycle=2, offsets=0).fmtc.resample(w=src.width, h=2 * src.height, kernel="spline64", sx=0, sy=0.25, sw=src.width, sh=src.height).fmtc.bitdepth(bits=8)
        o = core.std.SelectEvery(src, cycle=2, offsets=1).fmtc.resample(w=src.width, h=2 * src.height, kernel="spline64", sx=0, sy=-0.25, sw=src.width, sh=src.height).fmtc.bitdepth(bits=8)
        return core.std.Interleave(clips=[e, o])
    if src.format == 'YUV420P8':
        return bob(src)
    y = core.std.ShufflePlanes(clips=src, planes=0, colorfamily=vs.GRAY)
    u = core.std.ShufflePlanes(clips=src, planes=1, colorfamily=vs.GRAY)
    v = core.std.ShufflePlanes(clips=src, planes=2, colorfamily=vs.GRAY)
    if process_chroma:
        return core.std.ShufflePlanes([y, bob(u), bob(v)], planes=[0, 0, 0], colorfamily=vs.YUV)
    return core.std.ShufflePlanes([y, core.std.SelectEvery(u, cycle=1, offsets=[0, 0]), core.std.SelectEvery(v, cycle=1, offsets=[0, 0])], planes=[0, 0, 0], colorfamily=vs.YUV)


def pointbob(src):
    src = src.std.SeparateFields(True)[::2]
    return core.resize.Point(src, src.width, 2*src.height)


def median3(a, b, c, grey=True):
    return core.std.Interleave([a, b, c]).rgvs.Clense(planes=[0, 1, 2] if grey is True else 0).std.SelectEvery(cycle=3, offsets=1)


def crossfieldrepair(clip, sw=2, sh=2, bobbed_clip=None, chroma=True):
    if (sw < 0 and sh < 0) or sw < 0 or sh < 0:
        raise ValueError("crossfieldrepair: sw/sh cannot be a negative integers")
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("crossfieldrepair:\"clip\" not a clip")
    if bobbed_clip is None:
        bobbed_clip = spline64bob(clip, process_chroma=chroma)
    bob_ex = haf.mt_expand_multi(bobbed_clip, planes=1, sw=sw, sh=sh)
    bob_in = haf.mt_inpand_multi(bobbed_clip, planes=1, sw=sw, sh=sh)
    if sw == 1 and sh == 1:
        e = core.std.SelectEvery(bobbed_clip, cycle=2, offsets=0)
        o = core.std.SelectEvery(bobbed_clip, cycle=2, offsets=1)
        re = core.rgvs.Repair(bobbed_clip, e, mode=[1])
        ro = core.rgvs.Repair(bobbed_clip, o, mode=[1])
    else:
        ex = core.std.SelectEvery(bob_ex, cycle=2, offsets=0)
        ox = core.std.SelectEvery(bob_ex, cycle=2, offsets=1)
        ei = core.std.SelectEvery(bob_in, cycle=2, offsets=0)
        oi = core.std.SelectEvery(bob_in, cycle=2, offsets=1)
        re = median3(bobbed_clip, ex, ox, False)
        ro = median3(bobbed_clip, ei, oi, False)
    res = core.std.Interleave(clips=[re, ro])
    res = res.std.SeparateFields(True)[::2]
    res = core.std.SelectEvery(res, cycle=4, offsets=[2, 1])
    res = res.std.DoubleWeave()[::2]
    return res


def maxyuv(c):
    y = c.resize.Bicubic(width=c.width, height=c.height, format=vs.YUV420P8)
    u = core.resize.Bicubic(core.std.ShufflePlanes(clips=c, planes=1, colorfamily=vs.GRAY), c.width, c.height, vs.YUV420P8)
    v = core.resize.Bicubic(core.std.ShufflePlanes(clips=c, planes=2, colorfamily=vs.GRAY), c.width, c.height, vs.YUV420P8)
    w = c.width
    h = c.height

    yc = core.resize.Bilinear(y, u.width, u.height)
    ls = core.std.Expr([y, core.resize.Bilinear(u, w, h)], expr=['x y max']).std.Expr([y, core.resize.Bilinear(v, w, h)], expr=['x y max'])
    cs = core.std.Expr([yc, u], expr=['x y max']).std.Expr([yc, v], expr=['x y max'])
    return core.std.ShufflePlanes([cs, cs, ls], planes=[0, 0, 0], colorfamily=vs.YUV)


def mpeg2stinx(src, mode=1, sw=1, sh=1, contra=True, blurv=0.0, sstr=2.0, scl=0.25, dither=False, order=0, diffscl=None):
    """
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
    """

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
            bobbed = core.nnedi3.nnedi3(src, field=3)

        if order == 0:
            return bobbed
        elif order == 1:
            return core.std.SelectEvery(core.yadifmod.Yadifmod(src, order=0, mode=3, edeint=core.std.SelectEvery(bobbed, 2, [1, 0])).selectevery(2, 1, 0), 2, [1, 0])
        elif order == 2:
            return core.yadifmod.Yadifmod(src, order=1, mode=3, edeint=bobbed)
        raise ValueError('mpeg2stinx.deint: order can only be 0, 1, and 2')

    def templimit(c, flt, ref, diffscl):
        adj = ref.std.SelectEvery(2, [0, 1])
        diff = core.std.Expr(core.std.SelectEvery(c, 3, [0, 1]), adj, ["x y - abs"])
        diff = core.std.SeparateFields(True)[::2]
        diff = maxyuv(diff)
        diff2 = core.std.Expr(core.std.SelectEvery(diff, 4, [0, 1]), core.std.SelectEvery(diff, 4, [2, 3]), expr=["x y min"])
        diff2 = haf.mt_expand_multi(diff2, sw=2, sh=1, planes=0)
        diff2 = diff2.std.DoubleWeave()[::2]
        a = core.misc.AverageFrames(clips=[c, diff2], weights=[1, -diffscl])
        b = core.misc.AverageFrames(clips=[c, diff2], weights=[1, diffscl])
        return median3(a, b, flt)

    a = crossfieldrepair(src, sw=sw, sh=sh, bobbed_clip=deint(src, mode, order))
    if diffscl is not None:
        a = templimit(src, a, src, diffscl)
    b = crossfieldrepair(a, sw=sw, sh=sh, bobbed_clip=deint(a, mode, order))
    if diffscl is not None:
        b = templimit(a, b, src, diffscl)

    if dither:
        dit = core.misc.AverageFrames(clips=[a, b], weights=[0.5, 0.5])
        dit = mvf.Depth(dit, dither=4)
    else:
        dit = core.misc.AverageFrames(clips=[a, b], weights=[1, 1])

    if blurv > 0.0:
        nuked = core.std.BoxBlur(dit, hradius=1, vradius=blurv)
    else:
        nuked = dit

    nukedd = core.std.MakeDiff(src, nuked, [0, 1, 2])

    sharp = core.std.Expr(nuked, core.std.BoxBlur(nuked, hradius=0, vradius=1).std.BoxBlur(nuked, hradius=0, vradius=0), expr=["x x y - {} * +".format(sstr)])
    sharpd = core.std.Expr(nuked, core.std.BoxBlur(nuked, hradius=0, vradius=1).std.BoxBlur(nuked, hradius=0, vradius=0), expr=["x y - {} * 128 +".format(sstr)])
    limd = core.std.Expr(sharpd, nukedd, expr=[f"x 128 - y 128 - * 0 < {scl} 1 ? x 128 - abs y 128 - abs < x y ? 128 - * 128 +"])

    if scl == 0:
        last = median3(nuked, sharp, src)
    else:
        last = core.std.MergeDiff(nuked, limd, [0, 1, 2])

    if contra:
        return last
    return nuked

