from typing import cast

from opencosmo.spatial.region import BoxRegion, BoxSize, Point3d


def Box(center: Point3d, width: float | BoxSize):
    if isinstance(width, float) or isinstance(width, int):
        width = (width, width, width)

    halfwidth = cast(BoxSize, tuple(float(w / 2) for w in width))
    return BoxRegion(center, halfwidth)
