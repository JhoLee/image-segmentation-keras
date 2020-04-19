import numpy as np
import PIL.Image
import matplotlib.pyplot as plt


def resize_image_basic(img: np.array, height: int, width: int):
    if len(img.shape) == 3:
        o_height, o_width, channel = img.shape
    else:
        o_height, o_width = img.shpae
        channel = 0

    gap_height = height - o_height
    gap_width = width - o_width

    if gap_height != 0:
        gap_height = [gap_height // 2, gap_height // 2] \
            if gap_height % 2 == 0 \
            else [gap_height // 2 + 1, gap_height // 2]
    else:
        gap_height = [0, 0]
    if gap_width != 0:
        gap_width = [gap_width // 2, gap_width // 2] \
            if gap_width % 2 == 0 \
            else [gap_width // 2 + 1, gap_width // 2]
    else:
        gap_width = [0, 0]

    print(gap_height, gap_width)

    pad = []
    crop = []
    if gap_height[0] > 0:
        pad.append(gap_height)
        crop.append([0, len(img)])
    else:
        pad.append([0, 0])
        crop.append([gap_height[0] * -1, gap_height[1]])
    if gap_width[0] > 0:
        pad.append(gap_width)
        crop.append([0, len(img[0])])
    else:
        pad.append([0, 0])
        crop.append([gap_width[0] * -1, gap_width[1]])

    if channel > 0:
        resized_img = img[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1], :]
        resized_img = np.pad(resized_img, (pad[0], pad[1], [0, 0]), 'constant', constant_values=0)
    else:
        resized_img = img[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]
        resized_img = np.pad(resized_img, (pad[0], pad[1]), 'constant', constant_values=0)

    return resized_img


