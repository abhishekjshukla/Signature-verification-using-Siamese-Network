
from PIL import Image
import cv2
import numpy as np
from scipy import ndimage
def normalize_image(img, size=(840, 1360)):

    max_r, max_c = size

    blur_radius = 2
    blurred_image = ndimage.gaussian_filter(img, blur_radius)

    
    
    threshold, binarized_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    
    r, c = np.where(binarized_image == 0)
    r_center = int(r.mean() - r.min())
    c_center = int(c.mean() - c.min())

    
    cropped = img[r.min(): r.max(), c.min(): c.max()]

    
    img_r, img_c = cropped.shape

    r_start = max_r // 2 - r_center
    c_start = max_c // 2 - c_center

    
    
    
    if img_r > max_r:
        print ('Warning: cropping image. The signature should be smaller than the canvas size')
        r_start = 0
        difference = img_r - max_r
        crop_start = difference // 2
        cropped = cropped[crop_start:crop_start + max_r, :]
        img_r = max_r
    else:
        extra_r = (r_start + img_r) - max_r
        
        if extra_r > 0:
            r_start -= extra_r
        if r_start < 0:
            r_start = 0

    
    if img_c > max_c:
        print ('Warning: cropping image. The signature should be smaller than the canvas size')
        c_start = 0
        difference = img_c - max_c
        crop_start = difference // 2
        cropped = cropped[:, crop_start:crop_start + max_c]
        img_c = max_c
    else:
        extra_c = (c_start + img_c) - max_c
        if extra_c > 0:
            c_start -= extra_c
        if c_start < 0:
            c_start = 0

    normalized_image = np.ones((max_r, max_c), dtype=np.uint8) * 255
    
    normalized_image[r_start:r_start + img_r, c_start:c_start + img_c] = cropped

    
    normalized_image[normalized_image > threshold] = 255

    return normalized_image


def remove_background(img):
        """ Remove noise using OTSU's method.
        :param img: The image to be processed
        :return: The normalized image
        """

        img = img.astype(np.uint8)
        
        
        threshold, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img[img > threshold] = 255

        return img