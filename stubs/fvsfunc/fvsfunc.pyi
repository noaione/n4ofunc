from typing import Literal, Optional, Union, overload
import vapoursynth as vs

DitherType = Literal["none", "ordered", "random", "error_diffusion"]
RangeStr = Literal["limited", "full"]
RangeInt = Literal[0, 1]
Range = Union[RangeStr, RangeInt]

@overload
def Depth(
    src: vs.VideoNode,
    bits: int,
) -> vs.VideoNode: ...
@overload
def Depth(
    src: vs.VideoNode,
    bits: int,
    dither_type: DitherType = "error_diffusion",
    range: Optional[Range] = None,
    range_in: Optional[Range] = None,
) -> vs.VideoNode: ...
