# augmix : https://github.com/google-research/augmix

from PIL import Image
from PIL import ImageOps
import numpy as np

def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
    Returns:
    An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
    Returns:
    A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.

def sample_level(n):
    return np.random.uniform(low=0.1, high=n)

def autocontrast(pil_img, _, __):
    return ImageOps.autocontrast(pil_img)

def equalize(pil_img, _, __):
    return ImageOps.equalize(pil_img)

def posterize(pil_img, level, __):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)

def rotate(pil_img, level, __):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)

def solarize(pil_img, level, __):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)

def shear_x(pil_img, level, SIZE):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((SIZE, SIZE),
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)

def shear_y(pil_img, level, SIZE):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((SIZE, SIZE),
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)

def translate_x(pil_img, level, SIZE):
    level = int_parameter(sample_level(level), SIZE / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((SIZE, SIZE),
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level, SIZE):
    level = int_parameter(sample_level(level), SIZE / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((SIZE, SIZE),
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)

def normalize(image, MEAN, STD):
    """Normalize input image channel-wise to zero mean and unit variance."""
    mean, std = np.array(MEAN), np.array(STD)
    image = (image - mean) / std
    return image


def apply_op(image, op, severity, SIZE):
    image = np.clip(image * 255., 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(image)  # Convert to PIL.Image
    pil_img = op(pil_img, severity, SIZE)
    return np.asarray(pil_img) / 255.

def augment_and_mix(image, severity=-1, width=3, depth=-1, alpha=1.):
    """Perform AugMix augmentations and compute mixture.
    Args:
    image: Raw input image as float32 np.ndarray of shape (h, w, c)
    severity: Severity of underlying augmentation operators (between 1 to 10).
      if set < 0, random number between [1, 3] is chosen.
    width: Width of augmentation chain
    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
      from [1, 3]
    alpha: Probability coefficient for Beta and Dirichlet distributions.
    Returns:
    mixed: Augmented and mixed image.
  """
    augmentations = [autocontrast, equalize, posterize, rotate, 
                    solarize, shear_x, shear_y, translate_x, translate_y]

    ws = np.float32(np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))
    mix = np.zeros_like(image)
    if severity < 0:
        severity = np.random.randint(1, 4)
    for i in range(width):
        image_aug = image.copy()
        depth = depth if depth > 0 else np.random.randint(1, 4)
        
        for _ in range(depth):
            op = np.random.choice(augmentations)
            image_aug = apply_op(image_aug, op, severity, image_aug.shape[0])
        # mix = np.add(mix, ws[i] * normalize(image_aug, MEAN, STD), 
        #              out=mix, casting="unsafe")
        mix += ws[i] * image_aug

    mixed = (1 - m) * image + m * mix
    # mixed = (1 - m) * normalize(image, MEAN, STD) + m * mix
    return mixed

# test
#print(augment_and_mix(np.random.uniform(size=(5, 5))))