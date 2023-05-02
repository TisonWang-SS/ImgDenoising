import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import random

# data augmentation
def get_train_transforms(size):
    return albumentations.Compose(
        [
            albumentations.Resize(size, size),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=[-90, 90], p=0.5),
            # albumentations.RandomBrightnessContrast(),
            # albumentations.ShiftScaleRotate(
            #     shift_limit=0.25, scale_limit=0.1, rotate_limit=0
            # ),
            # albumentations.Normalize(
            #     [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
            #     max_pixel_value=255.0, always_apply=True
            # ),
            ToTensorV2(p=1.0),
        ]
    )

def get_valid_transforms():
    return albumentations.Compose(
        [
            # albumentations.Resize(size, size),
            # albumentations.Normalize(
            #     [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
            #     max_pixel_value=255.0, always_apply=True
            # ),
            ToTensorV2(p=1.0)
        ]
    )

def data_augmentation(image, mode):
    '''
    Performs data augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    '''
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out

def inverse_data_augmentation(image, mode):
    '''
    Performs inverse data augmentation of the input image
    '''
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        out = np.flipud(image)
    elif mode == 2:
        out = np.rot90(image, axes=(1,0))
    elif mode == 3:
        out = np.flipud(image)
        out = np.rot90(out, axes=(1,0))
    elif mode == 4:
        out = np.rot90(image, k=2, axes=(1,0))
    elif mode == 5:
        out = np.flipud(image)
        out = np.rot90(out, k=2, axes=(1,0))
    elif mode == 6:
        out = np.rot90(image, k=3, axes=(1,0))
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.flipud(image)
        out = np.rot90(out, k=3, axes=(1,0))
    else:
        raise Exception('Invalid choice of image transformation')

    return out

def random_augmentation(*args):
    out = []
    if random.randint(0,1) == 1:
        flag_aug = random.randint(1,7)
        for data in args:
            out.append(data_augmentation(data, flag_aug).copy())
    else:
        for data in args:
            out.append(data)
    return out