"""
MIT License

Copyright (c) 2020-present noaione

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import shutil
import sys
from functools import partial
from pathlib import Path
from typing import Dict, Generator, List, NamedTuple, NoReturn, Optional, Tuple

import vapoursynth as vs
from vsutil import get_w, get_y

__all__ = (
    "check_difference",
    "save_difference",
    "stack_compare",
    "interleave_compare",
    "compare",
)
core = vs.core


class FrameDiff(NamedTuple):
    frame: vs.VideoNode
    number: int
    difference: float


def _pad_video_length(clip_a: vs.VideoNode, clip_b: vs.VideoNode) -> Tuple[vs.VideoNode, vs.VideoNode]:
    clip_a_length = clip_a.num_frames
    clip_b_length = clip_b.num_frames
    if clip_a_length == clip_b_length:
        return clip_a, clip_b
    elif clip_a_length > clip_b_length:
        src_add = clip_a_length - clip_b_length
        clip_b = clip_b + (clip_b[-1] * src_add)
    elif clip_b_length > clip_a_length:
        src_add = clip_b_length - clip_a_length
        clip_a = clip_a + (clip_a[-1] * src_add)
    return clip_a, clip_b


def _preprocess_clips(clip_a: vs.VideoNode, clip_b: vs.VideoNode) -> Tuple[vs.VideoNode, vs.VideoNode]:
    if not isinstance(clip_a, vs.VideoNode):
        raise TypeError("clip_a must be a clip")
    if not isinstance(clip_b, vs.VideoNode):
        raise TypeError("clip_b must be a clip")
    clipa_cf = clip_a.format.color_family
    clipb_cf = clip_b.format.color_family
    clipa_bits = clip_a.format.bits_per_sample
    clipb_bits = clip_b.format.bits_per_sample

    clip_a, clip_b = _pad_video_length(clip_a, clip_b)

    if clipa_cf != vs.RGB:
        clip_a = clip_a.resize.Point(format=vs.RGBS, matrix_in_s="709")
    if clipb_cf != vs.RGB:
        clip_b = clip_b.resize.Point(format=vs.RGBS, matrix_in_s="709")

    if clipa_bits != 8:
        clip_a = clip_a.fmtc.bitdepth(bits=8)
    if clipb_bits != 8:
        clip_b = clip_b.fmtc.bitdepth(bits=8)
    return clip_a, clip_b


def _frame_yielder(
    clip_a: vs.VideoNode, clip_b: vs.VideoNode, threshold: float = 0.1
) -> Generator[Tuple[int, vs.RawFrame], None, None]:
    clip_a_gray = clip_a.std.ShufflePlanes(0, vs.GRAY)
    clip_b_gray = clip_b.std.ShufflePlanes(0, vs.GRAY)

    frame: vs.RawFrame
    for num, frame in enumerate(core.std.PlaneStats(clip_a_gray, clip_b_gray).frames()):
        if frame.props["PlaneStatsDiff"] >= threshold:
            yield num, frame


def check_difference(
    clip_a: vs.VideoNode,
    clip_b: vs.VideoNode,
    threshold: float = 0.1,
) -> NoReturn:
    if not hasattr(sys, "argv"):  # Simple check if script are opened via VSEdit
        raise Exception(
            "check_difference: please run this vpy script via command-line (ex: python3 ./script.vpy)"
        )

    clip_a, clip_b = _preprocess_clips(clip_a, clip_b)
    last_known_diff = -1
    known_diff = 0
    try:
        for num, _ in _frame_yielder(clip_a, clip_b, threshold):
            if last_known_diff != num:
                print(f"check_difference: Frame {num} is different")
                known_diff += 1
            last_known_diff = num + 1
    except KeyboardInterrupt:
        print("check_difference: Process interrupted")
        exit(1)

    if known_diff == 0:
        print(f"check_difference: No difference found (threshold: {threshold})")
    exit(0)


def save_difference(
    clip_a: vs.VideoNode,
    clip_b: vs.VideoNode,
    threshold: float = 0.1,
    output_filename: Tuple[str, str] = ["src1", "src2"],
) -> NoReturn:
    if not hasattr(sys, "argv"):  # Simple check if script are opened via VSEdit
        raise Exception(
            "save_difference: please run this vpy script via command-line (ex: python3 ./script.vpy)"
        )

    if len(output_filename) != 2:
        raise Exception("save_difference: output_filename must be a tuple of two strings")
    clip_a, clip_b = _preprocess_clips(clip_a, clip_b)
    fn_a, fn_b = output_filename

    # Get current directory of the file
    filename = Path(sys.argv[0]).resolve()
    current_dir = filename.parent

    only_fn = filename.name.split(".", 1)[0]

    target_dir = current_dir / f"{only_fn}_frame_difference"
    target_dir.mkdir(exist_ok=True)

    differences_data: Dict[str, FrameDiff] = {}
    known_diff = 0
    last_known_diff = -1
    try:
        for num, frame in _frame_yielder(clip_a, clip_b, threshold):
            if last_known_diff != num:
                differences_data[f"{known_diff}_{fn_a}"] = FrameDiff(
                    frame=clip_a[num],
                    number=num,
                    difference=frame.props["PlaneStatsDiff"],
                )
                known_diff += 1
            last_known_diff = num + 1
    except KeyboardInterrupt:
        print("save_difference: Process interrupted")
        exit(1)

    if known_diff == 0:
        print(f"check_difference: No difference found (threshold: {threshold})")
        shutil.rmtree(str(target_dir))
        exit(0)

    print(f"save_difference: {known_diff} differences found, saving images...")
    try:
        for filename, frame_info in differences_data.items():
            print(f"save_difference: saving frame: {frame_info.number} ({frame_info.difference})")
            actual_target = target_dir / f"{filename} (%05d).png"
            out: vs.VideoNode = core.imwri.Write(
                frame_info.frame, filename=str(actual_target), firstnum=frame_info.number
            )
            out.get_frame(0)
    except KeyboardInterrupt:
        print("save_difference: Process interrupted")
        exit(1)
    exit(0)


def stack_compare(
    clips: List[vs.VideoNode],
    height: Optional[int] = None,
    identity: bool = False,
    max_vertical_stack: int = 2,
    interleave_only: bool = False,
):
    """
    Stack/interleave compare multiple clips.
    Probably inspired by LightArrowsEXE ``stack_compare`` function.

    Clips are stacked like this:
    -------------
    | A | C | E |
    -------------
    | B | D | F |
    ------------- -- (For max_vertical_stack = 2)
    etc...

    If clips total are missing some, it'll add an extra BlankClip.
    Formula: `multiples_of_max_vertical_stack[i] <= clip_total <= multiples_of_max_vertical_stack[i + 1]`

    If one of the clips only have `Y` plane, all other clips will be changed to use only 1 plane
    The total vertical clips can be modified using `max_vertical_stack`

    Parameters
    ----------
    clips: :class:`List[VideoNode]`
        A collection of clips or sources to compare.
    height: :class:`Optional[int]`
        Resize the stacked compare into a new height.
        If ``interleave_only`` is ``True``, this will be ignored.
    identity: :class:`bool`
        If ``True``, there will be numbering to identify each clips.
        If ``interleave_only`` is ``True``, this will be ignored.
    max_vertical_stack: :class:`int`
        The maximum number of clips to stack vertically.
        If ``interleave_only`` is ``True``, this will be ignored.
    interleave_only: :class:`bool`
        If ``True``, the output will be an interleaved comparision.

    Returns
    -------
    :class:`VideoNode`
        A stacked/interleaved compare of the clips.
    """
    the_string = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789abcefghijklmnopqrstuvwxyz"
    if len(clips) < 2:
        raise ValueError("stack_compare: please provide 2 or more clips.")

    def _fallback_str(num: int) -> str:
        try:
            return the_string[num]
        except IndexError:
            return f"Extra{num + 1}"

    def _generate_ident(clip_index: int, src_w: int, src_h: int) -> str:
        gen = r"{\an7\b1\bord5\c&H00FFFF\pos"
        gen += "({w}, {h})".format(w=25 * (src_w / 1920), h=25 * (src_h / 1080))
        gen += r"\fs" + "{0}".format(60 * (src_h / 1080)) + r"}"
        gen += "Clip {0}".format(_fallback_str(clip_index))
        return gen

    # Check for luma only clip
    only_use_luma = False
    for clip in clips:
        if clip.format.num_planes == 1:
            only_use_luma = True
            break

    if interleave_only:
        if only_use_luma:
            clips = [get_y(clip) for clip in clips]

        # Set identity
        if identity:
            clips = [
                clip.sub.Subtitle(
                    _generate_ident(
                        idx,
                        clip.width,
                        clip.height,
                    )
                )
                for idx, clip in enumerate(clips)
            ]
        return core.std.Interleave(clips, mismatch=True)

    def _calculate_needed_clip(max_vert: int, clip_total: int) -> int:
        multiples_of = list(range(max_vert, (clip_total + 1) * max_vert, max_vert))
        multiples_of_total = len(multiples_of)
        max_needed = None
        for i in range(multiples_of_total):
            if i + 1 == multiples_of_total - 1:
                break
            if multiples_of[i] <= clip_total <= multiples_of[i + 1]:
                max_needed = multiples_of[i + 1]
                break
        return max_needed

    # Set YUV video to Y video if only_use_luma.
    if only_use_luma:
        clips = [get_y(clip) for clip in clips]

    if identity:
        clips = [
            clip.sub.Subtitle(_generate_ident(ind, clip.width, clip.height)) for ind, clip in enumerate(clips)
        ]

    # Find needed clip for current `max_vertical_stack`.
    if len(clips) != max_vertical_stack:
        needed_clip = _calculate_needed_clip(max_vertical_stack, len(clips))
        f_clip = clips[0]
        for _ in range(needed_clip - len(clips)):
            clips.append(
                core.std.BlankClip(f_clip).sub.Subtitle(
                    r"{\an5\fs120\b1\pos("
                    + "{},{}".format(f_clip.width / 2, f_clip.height / 2)
                    + r")}BlankClip Pad\N(Ignore)"
                )
            )

    # Split into chunks of `max_vertical_stack` and StackVertical it.
    # Input: [A, B, C, D, E, F, G, H]
    # Output: [[A, B], [C, D], [E, F], [G, H]]
    clips = [
        core.std.StackVertical(clips[i : i + max_vertical_stack])
        for i in range(0, len(clips), max_vertical_stack)
    ]
    final_clip = core.std.StackHorizontal(clips) if len(clips) > 1 else clips[0]
    if height:
        if height != final_clip.height:
            ar = final_clip.width / final_clip.height
            final_clip = final_clip.resize.Bicubic(
                get_w(height, ar),
                height,
            )
    return final_clip


interleave_compare = partial(stack_compare, interleave_only=True)
compare = stack_compare
