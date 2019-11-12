from functools import partial

import nnedi3_rpow2 as edi
import fvsfunc as fvf
import havsfunc as haf
import kagefunc as kgf
import mvsfunc as mvf
import vapoursynth as vs

from vsutil import get_y, is_image, iterate, split, get_w, insert_clip

"""
Something something N4O vapoursynth function something something
Refer to docstring for more info

Main Function:
    save_difference
    adaptive_degrain2
    masked_descale
    source
    adaptive_scaling
    SimpleFrameReplace
    better_planes
    better_frame
    recursive_apply_mask

Utility:
    is_extension
    register_f
    antiedgemask
    simple_native_mask
"""

core = vs.core

#helper
def is_extension(x: str, y: str) -> bool:
    """
    Return a boolean if extension are the same or not
    `x` are lowered/downcased

    :param x: str: A full filename with extension
    :param y: str: extension format in str (without dot)

    :return: bool
    """
    return x.lower()[x.lower().rfind('.'):] == y


def register_f(c: vs.VideoNode, yuv444=False) -> vs.VideoNode.format:
    """
    Return a registered new format

    :param c: vapoursynth.VideoNode: VideoNode object
    :param yuv444: bool: Do you want to register it as 4:4:4 subsampling or not

    :return: vapoursynth.VideoNode.format
    """
    return core.register_format(
        c.format.color_family,
        c.format.sample_type,
        c.format.bits_per_sample,
        0 if yuv444 else c.format.subsampling_w,
        0 if yuv444 else c.format.subsampling_w
    )


def save_difference(src1: vs.VideoNode, src2: vs.VideoNode, threshold: float = 0.1, output_fn: list = ['src1', 'src2'], check_only=False):
    """
    n4ofunc.save_difference

    Save a difference between src1 and src2
    Useful for comparing between TV and BD

    :param src1: vapoursynth.VideoNode: Video Source 1 as the "old" video
    :param src2: vapoursynth.VideoNode: Video Source 2 as the "new" video
    :param threshold: float: Luma threshold between src1 and src2
    :param output_fn: list: Output name for source (default are `src1` and `src2`)
    :param check_only: bool: Do a check only without saving image (will output if difference detected).
    """
    import os
    import shutil

    src1_cf = src1.format.color_family
    src2_cf = src2.format.color_family
    src1_bits = src1.format.bits_per_sample
    src2_bits = src2.format.bits_per_sample

    if src1.num_frames != src2.num_frames:
        raise ValueError('save_difference: src1 and src2 total frames are not the same')
    if len(output_fn) < 2:
        raise ValueError('save_difference: `output_fn` need a minimum of 2 name')
    out_fn1, out_fn2 = output_fn[:2]

    cwd = os.getcwd()
    dirsave = cwd + '\\frame_difference'
    print('[@] save_difference: Starting process')

    if not os.path.isdir(dirsave) and not check_only:
        os.mkdir(dirsave)

    if src1_cf != vs.RGB:
        src1 = src1.resize.Point(format=vs.RGBS, matrix_in_s='709')
    if src2_cf != vs.RGB:
        src2 = src2.resize.Point(format=vs.RGBS, matrix_in_s='709')

    if src1_bits != 8:
        src1 = src1.fmtc.bitdepth(bits=8)
    if src2_bits != 8:
        src2 = src2.fmtc.bitdepth(bits=8)

    src1_gray = src1.std.ShufflePlanes(0, vs.GRAY)
    src2_gray = src2.std.ShufflePlanes(0, vs.GRAY)

    n = 0
    dataset = dict()
    try:
        for i, f in enumerate(core.std.PlaneStats(src1_gray, src2_gray).frames()):
            print('[@] save_difference: Processing Frame {}/{} ({})'.format(i, src1_gray.num_frames, f.props["PlaneStatsDiff"]), end='\r')
            if f.props["PlaneStatsDiff"] >= threshold:
                if check_only:
                    print('', end='\n')
                    print('[@] save_difference: Difference detected: Frame {}'.format(i))
                    print('', end='\n')
                else:
                    dataset['{n}_{fn}'.format(n=i, fn=out_fn1)] = [src1[i], i, f.props['PlaneStatsDiff']]
                    dataset['{n}_{fn}'.format(n=i, fn=out_fn2)] = [src2[i], i, f.props['PlaneStatsDiff']]
                n += 1
        print('', end='\n')
    except KeyboardInterrupt:
        print('', end='\n')
        print('[!!] CTRL+C Pressed, halting frame processing...')
    if n == 0:
        print('[@] save_difference: no significant difference found with current threshold.')
        if not check_only:
            shutil.rmtree(dirsave)
        return

    if check_only:
        return

    print('[@] save_difference: Saving image...')
    print('[#] Total found: {}'.format(n))
    try:
        for namae, clips in dataset.items():
            clip, frame, diff_amount = clips
            print('[@] Saving Frame: {} ({})'.format(namae, diff_amount))
            out = core.imwri.Write(clip, 'PNG', f"{dirsave}\\{namae} (%05d).png", firstnum=frame)
            out.get_frame(0)
    except KeyboardInterrupt:
        print('[!!] CTRL+C Pressed, stopping...')


def adaptive_degrain2(src, luma_scaling=None, kernel='smdegrain', area='light', iter_edge=0, show_mask=False, **degrain_args):
    """
    An adaptive degrainer that took kageru adaptive_grain function and apply a degrainer of choice

    Available degrain kernel are:
    - SMDegrain
    - KNLMeansCL
    - TNLMeansCL
    - BM3D
    - DFTTest

    :param src: vapoursynth.VideoNode: A VideoNode class
    :param luma_scaling: int: a luma scaling for kageru adaptive_grain mask
    :param kernel: str: A kernel that will be used for degraining
    :param area: str: Area that will be degrained (light area or dark area)
    :param iter_edge: int: Edge iteration that will make sure lineart not to get degrained too
    :param show_mask: bool: Show mask that will be used for degraining
    :param degrain_args: kwargs: A kernel kwargs that will be passed onto degrainer kernel arguments

    :return: vapoursynth.VideoNode
    """
    kernel_kwargs = {
        'smdegrain': ['tr', 'thSAD', 'thSADC', 'RefineMotion', 'contrasharp', 'CClip', 'interlaced', 'tff', 'plane', 'Globals',
                      'pel', 'subpixel', 'prefilter', 'mfilter', 'blksize', 'overlap', 'search', 'truemotion', 'MVglobal', 'dct',
                      'limit', 'limitc', 'thSCD1', 'thSCD2', 'chroma', 'hpad', 'vpad', 'Str', 'Amp'],
        'knlmeanscl': ['d', 'a', 's', 'h', 'channels', 'wmode', 'wref', 'rclip', 'device_type',
                       'device_id', 'ocl_x', 'ocl_y', 'ocl_r', 'info'],
        'tnlmeanscl': ['ax', 'ay', 'az', 'sx', 'sy', 'bx', 'by', 'a', 'h', 'ssd'],
        'bm3d': ['sigma', 'radius1', 'radius2', 'profile1', 'profile2', 'refine', 'pre', 'ref', 'psample',
                 'matrix', 'full', 'output', 'css', 'depth', 'sample', 'dither', 'useZ', 'prefer_props',
                 'ampo', 'ampn', 'dyn', 'staticnoise', 'cu_kernel', 'cu_taps', 'cu_a1', 'cu_a2', 'cu_cplace',
                 'cd_kernel', 'cd_taps', 'cd_a1', 'cd_a2', 'cd_cplace', 'cd_a1', 'cd_a2', 'cd_cplace',
                 'block_size1', 'block_step1', 'group_size1', 'bm_range1', 'bm_step1', 'ps_num1', 'ps_range1', 'ps_step1', 'th_mse1',
                 'block_size2', 'block_step2', 'group_size2', 'bm_range2', 'bm_step2', 'ps_num2', 'ps_range2', 'ps_step2', 'th_mse2', 'hard_thr'],
        'dfttest': ['ftype', 'sigma', 'sigma2', 'pmin', 'pmax', 'sbsize', 'smode', 'sosize', 'tbsize', 'tmode', 'tosize', 'swin', 'twin', 
                    'sbeta', 'tbeta', 'zmean', 'f0beta', 'nstring', 'sstring', 'ssx', 'ssy', 'sst', 'planes', 'opt']
    }

    valid_kernels = {
        "smd": "smdegrain",
        "bm3d": "bm3d",
        "dft": "dfttest",
        "knlm": "knlmeanscl",
        "tnlm": "tnlmeanscl",
        "knl": "knlmeanscl",
        "tnl": "tnlmeanscl",
        "smdegrain": "smdegrain",
        "knlmeanscl": "knlmeanscl",
        "tnlmeanscl": "tnlmeanscl",
        "dfttest": "dfttest"
    }

    if not isinstance(src, vs.VideoNode):
        raise ValueError("adaptive_degrain: 'src' must be a clip")
    if luma_scaling is None:
        luma_scaling = 30
    if area not in ['dark', 'light']:
        raise ValueError('adaptive_degrain: `area` can only be: `light` and `dark`')

    kernel = kernel.lower()

    if kernel not in valid_kernels:
        raise ValueError("adaptive_degrain: 'kernel' {} is not exist or not supported".format(kernel))

    kernel = valid_kernels[kernel]

    for arg in degrain_args:
        if arg not in kernel_kwargs[kernel]:
            raise ValueError('adaptive_degrain' + ": '" + arg + "' is not a valid argument for " + kernel)

    degrainfuncs = {
        'smdegrain': (lambda src: haf.SMDegrain(src, **degrain_args)),
        'bm3d': (lambda src: mvf.BM3D(src, **degrain_args)),
        'knlmeanscl': (lambda src: core.knlm.KNLMeansCL(src, **degrain_args)),
        'tnlmeanscl': (lambda src: core.tnlm.TNLMeans(src, **degrain_args)),
        'dfttest': (lambda src: core.dfttest.DFTTest(src, **degrain_args)),
    }

    adaptmask = kgf.adaptive_grain(src, luma_scaling=luma_scaling, show_mask=True)
    y_plane = get_y(src)

    if area == 'light':
        adaptmask = adaptmask.std.Invert()

    limitx = y_plane.std.Convolution([-1, -2, -1, 0, 0, 0, 1, 2, 1], saturate=False)
    limity = y_plane.std.Convolution([-1, 0, 1, -2, 0, 2, -1, 0, 1], saturate=False)
    limit = core.std.Expr([limitx, limity], 'x y max')
    limit = iterate(limit, core.std.Inflate, iter_edge)

    mask = core.std.Expr([adaptmask, limit], 'x y -')

    if show_mask:
        return mask

    fil = degrainfuncs[kernel](src)

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


def simple_native_mask(clip, descale_w, descale_h, blurh=1.5, blurv=1.5, iter_max=3):
    clip32 = fvf.Depth(clip, 32)
    y_32 = get_y(clip32)
    clip_bits = clip.format.bits_per_sample

    target_w = clip.width
    target_h = clip.height

    down = core.descale.Debicubic(y_32, descale_w, descale_h)
    up = core.resize.Bicubic(down, target_w, target_h)
    dmask = core.std.Expr([y_32, up], 'x y - abs 0.025 > 1 0 ?')
    dmask = iterate(dmask, core.std.Maximum, iter_max).std.BoxBlur(hradius=blurh, vradius=blurv)
    return core.resize.Bicubic(dmask, descale_w, descale_h).fmtc.bitdepth(bits=clip_bits)


def masked_descale(src: vs.VideoNode, target_w=None, target_h=None, kernel='bicubic', b=1/3, c=1/3, taps=3, yuv444=False, expandN=1, masked=True, show_mask=False) -> vs.VideoNode:
    """
    A masked descale that will descale everything but "native" 1080p content or some kind of it

    :param src: vapoursynth.VideoNode: A VideoNode object
    :param target_w: int: Target descale width resolution
    :param target_h: int: Target descale height resolution
    :param kernel: str: Kernel that will be used for descaling and downscaling "native" content
    :param b: float: parameter for bicubic kernel
    :param c: float: parameter for bicubic kernel
    :param taps: int: parameter for lanczos kernel
    :param yuv444: bool: Dither to 4:4:4 chroma subsampling
    :param expandN: int: Iteration for mask, using core.std.Maximum
    :param masked: bool: Use mask for descaling, to make sure "native" content are not descaled but downscaled
    :param show_mask: bool: Show mask that are used.
    """
    if not src:
        raise ValueError('src cannot be empty')
    if not target_w:
        raise ValueError('target_w cannot be empty')
    if not target_h:
        raise ValueError('target_h cannot be empty')

    expandN -= 1

    if expandN < 0:
        raise ValueError('expandN cannot be negative integer')

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

    if masked:
        video_mask = simple_native_mask(src, target_w, target_h, expandN)

        if show_mask:
            return video_mask
        return core.std.MaskedMerge(descale, nr_resize, video_mask)
    return descale


def source(src, lsmas=False, depth=False, trims=None, dither_yuv=True) -> vs.VideoNode:
    """
    Open Video or Image Source
    :param src: str: Video or Image source (format: string)
    :param lsmas: bool: force use lsmas (file with .m2ts extension will forced to use lsmas)
    :param depth: int: Dither video (Disable with False, default: False (Use original bitdepth))
    :param trims: list: Trim video (Integer + List type)
    :param dither_yuv: bool: Dither Image or Video to YUV subsample
    """
    def parse_trim_data(trim_data, video):
        a, b = trim_data

        if b == 0:
            b = video.num_frames
        else:
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

    src = Source(src, lsmas)(src)

    if dither_yuv and src.format.color_family != vs.YUV:
        src = core.resize.Point(src, format=vs.YUV420P8, matrix_s='709')

    if depth:
        src = fvf.Depth(src, depth)
    if trims:
        first, last = parse_trim_data(trims, src)
        return core.std.Trim(src, first, last)
    return src


def adaptive_scaling(clip: vs.VideoNode, target_w=None, target_h=None, descale_range=[], kernel='bicubic', b=1/3, c=1/3, taps=3, iter_max=3, rescale=True, show_native_res=False, show_mask=False, test_dmask=False):
    """
    n4ofunc.adaptive_scaling
    Descale within range and upscale it back to target_w and target_h
    If target are not defined, it will be using original resolution

    Written originally by kageru, modified by N4O.

    :param clip: vapoursynth.VideoNode: A VideoNode object
    :param taint target_w: Target upscaled width resolution
    :param target_h: int: Target upscaled width resolution
    :param descale_range: list: Descale range number in list
    :param kernel: str: Descaling kernel
    :param b: float: parameter for bicubic kernel
    :param c: float: parameter for bicubic kernel
    :param taps: int: parameter for lanczos kernel
    :param iter_max: int: Iteration for mask, using core.std.Maximum
    :param show_mask: bool: Show native mask that are used.
    """
    target_w = clip.width if target_w is None else target_w
    target_h = clip.height if target_h is None else target_h
    kernel = kernel.lower()
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
            return core.resize.Bicubic
        elif kernel == 'lanczos':
            return core.resize.Lanczos
        elif kernel == 'spline16':
            return core.resize.Spline16
        elif kernel == 'spline36':
            return core.resize.Spline36

    def VideoDescaler(b, c, taps, kernel):
        if kernel == 'bilinear':
            return core.descale.Debilinear
        elif kernel == 'bicubic':
            return partial(core.descale.Debicubic, b=b, c=c)
        elif kernel == 'lanczos':
            return partial(core.descale.Delanczos, taps=taps)
        elif kernel == 'spline16':
            return core.descale.Despline16
        elif kernel == 'spline36':
            return core.descale.Despline36

    def simple_descale(y: vs.VideoNode, h: int) -> tuple:
        down = global_clip_descaler(y, get_w(h), h)
        if rescale:
            up = global_clip_resizer(down, target_w, target_h)
        else:
            up = global_clip_resizer(down, y.width, y.height)
        diff = core.std.Expr([y, up], 'x y - abs').std.PlaneStats()
        return down, diff

    ref = clip
    ref_d = ref.format.bits_per_sample
    clip32 = fvf.Depth(clip, 32)
    y = get_y(clip32)
    global_clip_resizer = VideoResizer(b, c, taps, kernel)
    global_clip_descaler = VideoDescaler(b, c, taps, kernel)

    descale_listp = [simple_descale(y, h) for h in range(descale_range[0], descale_range[1])]
    descale_list = [a[0] for a in descale_listp]
    descale_props = [a[1] for a in descale_listp]

    if not rescale:
        y = global_clip_resizer(y, target_w, target_h)
        clip32 = global_clip_resizer(clip32, target_w, target_h)

    def select(n, descale_list, f):
        errors = [x.props.PlaneStatsAverage for x in f]
        y_deb = descale_list[errors.index(min(errors))]
        dmask = core.std.Expr([y, global_clip_resizer(y_deb, target_w, target_h)], 'x y - abs 0.025 > 1 0 ?').std.Maximum()
        y_deb16 = fvf.Depth(y_deb, 16)

        if rescale:
            y_scaled = edi.nnedi3_rpow2(y_deb16, nns=4, correct_shift=True, width=target_w, height=target_h).fmtc.bitdepth(bits=32)
        else:
            y_scaled = global_clip_resizer(y_deb16, target_w, target_h).fmtc.bitdepth(bits=32)
        dmask = global_clip_resizer(dmask, target_w, target_h)
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

    line_mask = global_clip_resizer(full, target_w, target_h)

    artifacts = core.misc.Hysteresis(global_clip_resizer(dmask, target_w, target_h, _format=vs.GRAYS), core.std.Expr([get_y(clip32).tcanny.TCanny(sigma=3), line_mask], 'x y min'))

    ret_raw = kgf.retinex_edgemask(ref)
    if not rescale:
        ret_raw = global_clip_resizer(ret_raw, target_w, target_h)
    ret = ret_raw.std.Binarize(30).rgvs.RemoveGrain(3)
    mask = core.std.Expr([iterate(artifacts, core.std.Maximum, iter_max), ret.resize.Point(_format=vs.GRAYS)], 'y x -').std.Binarize(0.4)
    mask = mask.std.Inflate().std.Convolution(matrix=[1] * 9).std.Convolution(matrix=[1] * 9)

    if show_mask:
        return mask

    merged = core.std.MaskedMerge(y, y_deb, mask)
    merged = core.std.ShufflePlanes([merged, clip32], [0, 1, 2], vs.YUV)
    return fvf.Depth(merged, ref_d)


def SimpleFrameReplace(src: vs.VideoNode, src_frame: int, target_frame: str) -> vs.VideoNode:
    """
    A simple frame replacing, useful for replacing black frame with other frame from the same video since I'm lazy
    :param src: vapoursynth.VideoNode: Video Source
    :param src_frame: int: Video Frame number as Source for replacing
    :param target_frame: str: Video Target frame to be replaced from src_frame
                              can be used as range, write it: `x-y`
    """
    src_fpsnum = src.fps.numerator
    src_fpsden = src.fps.denominator

    src_frame = src[src_frame]

    frame_range = target_frame.split('-')

    if len(frame_range) < 2:
        frame_range = [int(frame_range[0]), int(frame_range[0]) + 1]
    else:
        frame_range = [int(frame_range[0]), int(frame_range[1]) + 1]

    if frame_range[0] > frame_range[1]:
        raise ValueError('SimpleFrameReplace: `target_frame` last range number are bigger than the first one')

    src_frame = src_frame * (frame_range[1] - frame_range[0])

    pre = src[:frame_range[0]]
    post = src[frame_range[1]:]

    return pre + src_frame + post


def select_best(n, f, clist, pd):
    clip_data = []
    if pd == "BothAdd":
        for p in f:
            clip_data.append(p.props['PlaneStatsMax'] + p.props['PlaneStatsMin'])
    elif pd == "BothSubtract":
        for p in f:
            clip_data.append(p.props['PlaneStatsMax']-p.props['PlaneStatsMin'])
    else:
        for p in f:
            clip_data.append(p.props[pd])
    return clist[clip_data.index(max(clip_data))]


def better_planes(clips: vs.VideoNode, props="avg", show_info=False):
    """
    A naive function for picking the best planes from every frame from a list of video

    Every clips source planes are split into Y, U, and V
    Then using defined `props` they will be compared:
        - Avg: Highest Average PlaneStats
        - Min: Lowest Minimum PlaneStats
        - Max: Highest Maximum PlaneStats
        - Add: Value from subtracting PlaneStatsMax with PlaneStatsMin
    The best outcome plane will be returned

    `props` value must be:
    - For using PlaneStatsAverage as comparasion: "avg", "average", or "planestatsaverage"
    - For using PlaneStatsMin as comparasion: "min", "minimum", or "planestatsmin"
    - For using PlaneStatsMax as comparasion: "max", "maximum", or "planestatsmax"
    - For subtracting PlaneStatsMax with PlaneStatsMin as comparasion: "sub" or "subtract"
    - For combining value of PlaneStatsMax with PlaneStatsMin as comparasion: "add" or "addition"

    `show_info` are just showing what input will be used
    if it's True (bool) it will show it.
    You can customize it by passing a list of string to `show_info`

    :param clips: list: A list of vapoursynth.VideoNode object
    :param props: list or str: A list or string for comparing, if it's a list, the maximum is 3 (For Y, U, and V)
    :param show_info: bool: Show text for what source are used

    Example: 
    src = nao.better_planes(clips=[src_vbr, src_hevc, src_cbr], props=["max", "avg"], show_info=["AMZN VBR", "AMZN HEVC", "AMZN CBR"])
    src = nao.better_planes(clips=[src1, src2, src3], show_info=["CR", "Funi", "Abema"])
    src = nao.better_planes(clips=[src1, src2], props="max")
    """

    allowed_props = {
        "avg": "PlaneStatsAverage",
        "min": "PlaneStatsMin",
        "max": "PlaneStatsMax",
        "average": "PlaneStatsAverage",
        "minimum": "PlaneStatsMin",
        "maximum": "PlaneStatsMax",
        "planestatsaverage": "PlaneStatsAverage",
        "planestatsmin": "PlaneStatsMin",
        "planestatsmax": "PlaneStatsMax",
        "add": "BothAdd",
        "sub": "BothSubtract",
        "addition": "BothAdd",
        "subtract": "BothSubtract"
    }

    if isinstance(props, str):
        props = props.lower()
        if props in allowed_props:
            props_ = [allowed_props[props] for i in range(3)]
        else:
            raise ValueError("better_planes: `props` must be a `min` or `max` or `avg` or `add` or `sub`")
    elif isinstance(props, list):
        up_t = len(props)
        props_ = []
        if up_t == 1:
            t_props_ = [props[0] for i in range(3)]
        elif up_t == 2:
            t_props_ = [props[0], props[1], props[1]]
        elif up_t == 3:
            t_props_ = props
        elif up_t > 3:
            t_props_ = props[:3]
        for n, i in enumerate(t_props_):
            if i in allowed_props:
                props_.append(allowed_props[i.lower()])
            else:
                raise ValueError("better_planes: `props[{}]` must be a `min` or `max` or `avg` or `add` or `sub``".format(n))
    else:
        raise ValueError("better_planes: props must be a string or a list")

    clips1_ = []
    clips2_ = []
    clips3_ = []
    if isinstance(show_info, list):
        for n, clip in enumerate(clips):
            clips1_.append(core.text.Text(core.std.ShufflePlanes(clip, 0, vs.GRAY), "{} - Y ({})".format(show_info[n], props_[0]), 7))
            clips2_.append(core.text.Text(core.std.ShufflePlanes(clip, 1, vs.GRAY), "{} - U ({})".format(show_info[n], props_[1]), 8))
            clips3_.append(core.text.Text(core.std.ShufflePlanes(clip, 2, vs.GRAY), "{} - V ({})".format(show_info[n], props_[2]), 9))
    elif isinstance(show_info, bool) and show_info:
        for n, clip in enumerate(clips):
            clips1_.append(core.text.Text(core.std.ShufflePlanes(clip, 0, vs.GRAY), "Input {} - Y ({})".format(n+1, props_[0]), 7))
            clips2_.append(core.text.Text(core.std.ShufflePlanes(clip, 1, vs.GRAY), "Input {} - U ({})".format(n+1, props_[1]), 8))
            clips3_.append(core.text.Text(core.std.ShufflePlanes(clip, 2, vs.GRAY), "Input {} - V ({})".format(n+1, props_[2]), 9))
    else:
        for clip in clips:
            clips1_.append(core.std.ShufflePlanes(clip, 0, vs.GRAY))
            clips2_.append(core.std.ShufflePlanes(clip, 1, vs.GRAY))
            clips3_.append(core.std.ShufflePlanes(clip, 2, vs.GRAY))

    _clips_prop1 = []
    _clips_prop2 = []
    _clips_prop3 = []
    for clip in clips1_:
        _clips_prop1.append(clip.std.PlaneStats(plane=0))
    for clip in clips2_:
        _clips_prop2.append(clip.std.PlaneStats(plane=0))
    for clip in clips3_:
        _clips_prop3.append(clip.std.PlaneStats(plane=0))

    y_val = core.std.FrameEval(clips1_[0], partial(select_best, clist=clips1_, pd=props_[0]), prop_src=_clips_prop1)
    u_val = core.std.FrameEval(clips2_[0], partial(select_best, clist=clips2_, pd=props_[1]), prop_src=_clips_prop2)
    v_val = core.std.FrameEval(clips3_[0], partial(select_best, clist=clips3_, pd=props_[2]), prop_src=_clips_prop3)

    return core.std.ShufflePlanes([y_val, u_val, v_val], [0], vs.YUV)

# TODO: Maybe add chroma support or something
def better_frame(clips: vs.VideoNode, props="avg", show_info=False):
    """
    A naive function for picking the best frames from a list of video (Basically better_planes without checking chroma plane)

    This only check luma plane (Y plane) not like better_planes that check every plane
    Then using defined `props` they will be compared:
        - Avg: Highest Average PlaneStats
        - Min: Lowest Minimum PlaneStats
        - Max: Highest Maximum PlaneStats
        - Add: Value from subtracting PlaneStatsMax with PlaneStatsMin
    The best outcome plane will be returned

    `props` value must be:
    - For using PlaneStatsAverage as comparasion: "avg", "average", or "planestatsaverage"
    - For using PlaneStatsMin as comparasion: "min", "minimum", or "planestatsmin"
    - For using PlaneStatsMax as comparasion: "max", "maximum", or "planestatsmax"
    - For subtracting PlaneStatsMax with PlaneStatsMin as comparasion: "sub" or "subtract"
    - For combining value of PlaneStatsMax with PlaneStatsMin as comparasion: "add" or "addition"

    `show_info` are just showing what input will be used
    if it's True (bool) it will show it.
    You can customize it by passing a list of string to `show_info`

    :param clips: list: A list of vapoursynth.VideoNode object
    :param props: str: A string of allowed props
    :param show_info: bool: Show text for what source are used

    Example:
    src = nao.better_frame(clips=[src_vbr, src_hevc, src_cbr], props="add", show_info=["AMZN VBR", "AMZN HEVC", "AMZN CBR"])
    src = nao.better_frame(clips=[src1, src2, src3], show_info=["CR", "Funi", "Abema"])
    src = nao.better_frame(clips=[src1, src2], props="max")
    """

    allowed_props = {
        "avg": "PlaneStatsAverage",
        "min": "PlaneStatsMin",
        "max": "PlaneStatsMax",
        "average": "PlaneStatsAverage",
        "minimum": "PlaneStatsMin",
        "maximum": "PlaneStatsMax",
        "planestatsaverage": "PlaneStatsAverage",
        "planestatsmin": "PlaneStatsMin",
        "planestatsmax": "PlaneStatsMax",
        "add": "BothAdd",
        "sub": "BothSubtract",
        "addition": "BothAdd",
        "subtract": "BothSubtract"
    }

    if isinstance(props, str):
        props = props.lower()
        if props in allowed_props:
            props_ = allowed_props[props]
        else:
            raise ValueError("better_frame: `props` must be a `min` or `max` or `avg` or `add` or `sub`")
    else:
        raise ValueError("better_frame: props must be a string")

    clips_ = []
    if isinstance(show_info, list):
        for n, clip in enumerate(clips):
            clips_.append(core.text.Text(clip, "{} - ({})".format(show_info[n], props_), 7))
    elif isinstance(show_info, bool) and show_info:
        for n, clip in enumerate(clips):
            clips_.append(core.text.Text(clip, "Input {} - ({})".format(n+1, props_), 7))
    else:
        for clip in clips:
            clips_.append(clip)

    _clips_prop = []
    for clip in clips_:
        _clips_prop.append(clip.std.PlaneStats(plane=0))

    return core.std.FrameEval(clips_[1], partial(select_best, clist=clips_, pd=props_), prop_src=_clips_prop)


def recursive_apply_mask(src1, src2, mask_folder):
    """
    Recursively check `mask_folder` for an .png file
    After it found all of them, it will loop it and apply the mask

    Acceptable filename format:
    frameNum.png
    Example: 2500.png

    :param src1: vapoursynth.VideoNode: A VideoNode clip (Format must be the same as src2)
    :param src2: vapoursynth.VideoNode: A VideoNode clip (Format must be the same as src1)
    :param mask_folder: str: A folder path that contains the masks
    :return: src1
    """
    import os
    import glob

    imwri = core.imwri
    masks = glob.glob(mask_folder.rstrip('/').rstrip('\\') + '/*.png')
    for mask in masks:
        im = fvf.Depth(
            imwri.Read(mask).resize.Point(
                format=vs.GRAYS,
                matrix_s='709'
            ),
            src1.format.bits_per_sample).std.BoxBlur(
                hradius=3,
                vradius=3
            ).std.AssumeFPS(
                fpsnum=src1.fps.numerator, fpsden=src1.fps.denominator
        )

        frame = int(os.path.splitext(os.path.basename(mask))[0])
        src1N, src2N = src1[frame], src2[frame]

        src_masked = core.std.MaskedMerge(src1N, src2N, im)
        src1 = insert_clip(src1, src_masked, frame)

    return src1


src = source
descale = masked_descale
antiedge = antiedgemask
adaptive_bm3d = partial(adaptive_degrain2, kernel='bm3d')
adaptive_dfttest = partial(adaptive_degrain2, kernel='dfttest')
adaptive_knlm = partial(adaptive_degrain2, kernel='knlm')
adaptive_tnlm = partial(adaptive_degrain2, kernel='tnlm')
adaptive_smdegrain = partial(adaptive_degrain2, kernel='smd')
adaptive_rescale = partial(adaptive_scaling, rescale=True)
adaptive_descale = partial(adaptive_scaling, rescale=False)
check_diff = partial(save_difference, check_only=True)
sfr = SimpleFrameReplace
bplanes = better_planes
bframe = better_frame
