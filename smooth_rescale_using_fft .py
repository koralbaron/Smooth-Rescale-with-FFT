from typing import Tuple

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt


def read_image_as_gray_scale(path: str):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def display_and_save_image(img: np.ndarray, img_saving_name: str, saving_folder: str, cmp: str, saving_format: str = 'png'):
    plt.imshow(img, cmap=cmp), plt.title(img_saving_name)
    plt.savefig(os.path.join(saving_folder, f'{img_saving_name}.{saving_format}'))
    plt.show()


def get_smooth_rescaled_image_mask_and_magnitude_spectrum_using_fft(img_path: str, rescale_image_factor: int, is_ghost_mode: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray_img = read_image_as_gray_scale(img_path)
    dft = cv2.dft(np.float32(gray_img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft) if is_ghost_mode else dft
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    rows, cols, ch = dft_shift.shape
    mask = np.zeros((rows * rescale_image_factor, cols * rescale_image_factor, 2), np.float32)
    r = int(np.ceil(rows/2))
    c = int(np.ceil(cols/2))
    mask[0:r, 0:c] = dft_shift[0:r, 0:c]
    mask[-r+1:, 0:c] = dft_shift[r:, 0:c]
    mask[0:r, -c+1:] = dft_shift[0:r, c+1:]
    mask[-r+1:, -c+1:] = dft_shift[r:, c+1:]
    mask_magnitude_spectrum = 20 * np.log(cv2.magnitude(mask[:, :, 0], mask[:, :, 1]))
    mask_ishift = np.fft.ifftshift(mask) if is_ghost_mode else mask
    rescaled_image = cv2.idft(np.float32(mask_ishift))
    rescaled_image = cv2.magnitude(rescaled_image[:, :, 0], rescaled_image[:, :, 1])

    return rescaled_image, magnitude_spectrum, mask_magnitude_spectrum


def display_and_save_all_results(img_name_to_img_map: dict, output_folder: str, cmp: str = 'gray', saving_format: str = 'png'):
    for img_name, img in img_name_to_img_map.items():
        display_and_save_image(img, img_name, output_folder, cmp, saving_format)

def validate_folder_path(folder_path: str):
    if not os.path.isdir(folder_path):
        raise FileExistsError(f'File: "{folder_path}" not exists, please choose a valid path and run again')

def smooth_image_rescaling_and_save_results(imh_path: str, output_folder: str, rescale_image_factor: int, is_ghost_mode: bool = False,
                                            cmp: str = 'gray', saving_format: str = 'png'):
    validate_folder_path(output_folder)
    imgBack, magnitudeSpectrum, maskMagnitudeSpectrum = get_smooth_rescaled_image_mask_and_magnitude_spectrum_using_fft(imh_path, rescale_image_factor, is_ghost_mode)

    img_name_to_img_map = {'Rescaled_Img': imgBack, 'Magnitude_Spectrum_Img': magnitudeSpectrum, 'Mask_Magnitude_Spectrum_Img': maskMagnitudeSpectrum}
    display_and_save_all_results(img_name_to_img_map, output_folder, cmp, saving_format)
