
import numpy as np
import cv2

def get_random_kernel(low=1, high=5, size=2):
    structure = np.random.choice([cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS])
    kernel = cv2.getStructuringElement(structure, tuple(np.random.randint(low, high, size)))
    return kernel

def erode(img):
    img = cv2.erode(img, get_random_kernel(), iterations=1)
    return img

def dilate(img):
    img = cv2.dilate(img, get_random_kernel(), iterations=1)
    return img

def open_(img):
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, get_random_kernel())
    return img

def close(img):
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, get_random_kernel())
    return img
