import os
import os.path as osp

import cv2


BINARY_MODE = False

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
    (255, 0, 128),
    (0, 128, 255),
    (0, 255, 128),
    (128, 255, 255),
    (255, 128, 255),
    (255, 255, 128),
    (60, 180, 0),
    (180, 60, 0),
    (0, 60, 180),
    (0, 180, 60),
    (60, 0, 180),
    (180, 0, 60),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
]


def imshow_lanes(img,
                 lanes,
                 show=False,
                 out_file=None,
                 width=4,
                 binary_mode=BINARY_MODE):
    """Draw lane instances on an image."""
    lanes_xys = []
    for lane in lanes:
        xys = []
        for x, y in lane:
            if x < 0 or y < 0:
                continue
            xys.append((int(x), int(y)))
        if len(xys) > 0:
            lanes_xys.append(xys)

    if binary_mode:
        canvas = cv2.cvtColor(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            cv2.COLOR_GRAY2BGR
        ) * 0
    else:
        canvas = img.copy()

    valid_xys = [xys for xys in lanes_xys if len(xys) > 0]
    valid_xys.sort(key=lambda xys: xys[0][0] if xys else 0)

    for idx, xys in enumerate(lanes_xys):
        if len(xys) < 2:
            continue
        color = (255, 255, 255) if binary_mode else COLORS[idx % len(COLORS)]
        for i in range(1, len(xys)):
            cv2.line(canvas, xys[i - 1], xys[i], color, thickness=width)

    if show:
        cv2.imshow('view', canvas)
        cv2.waitKey(0)

    if out_file:
        out_dir = osp.dirname(out_file)
        if out_dir and not osp.exists(out_dir):
            os.makedirs(out_dir)
        cv2.imwrite(out_file, canvas)
    return canvas
