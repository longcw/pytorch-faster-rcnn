# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2


def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob


def _factor_closest(num, factor, is_ceil=True):
    """Returns the closest integer to `num` that is divisible by `factor`

    Actually, that's a lie. By default, we return the closest factor _greater_
    than the input. If, however, you set `it_ceil` to `False`, we return the
    closest factor _less_ than the input.
    """
    num = float(num) / factor
    num = np.ceil(num) if is_ceil else np.floor(num)
    return int(num) * factor


def pad_to_factor(im, factor=32):
    # Compute the padded image shape. Ensure it's divisible by factor.
    h, w = im.shape[:2]
    new_h, new_w = _factor_closest(h, factor), _factor_closest(w, factor)
    new_shape = [new_h, new_w] if im.ndim < 3 else [new_h, new_w, im.shape[-1]]

    # Pad the image.
    im_padded = np.full(new_shape, fill_value=0, dtype=im.dtype)
    im_padded[0:h, 0:w] = im

    return im_padded


def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    im_padded = pad_to_factor(im, 32)

    return im_padded, im_scale
