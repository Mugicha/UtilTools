import cv2
import os
import numpy as np
from tqdm import tqdm


class Image_processing():

    def smooth_histogram(self, _path: str):
        """
        ヒストグラム平滑化する機能。
        :param _path: 平滑化する画像のパス。
        :return:
        """
        img = cv2.imread(_path, 0)
        equ = cv2.equalizeHist(img)
        # res = np.hstack((img,equ))
        output_fol = '/'.join(_path.split('/')[0:-1]) + './smoothed'
        if not os.path.isdir(output_fol):
            os.mkdir(output_fol)
        output_img = ''.join(_path.split('/')[-1].split('.')[0:-1]) + '_smoothed.png'
        print(output_fol + '/' + output_img)
        cv2.imwrite(output_fol + '/' + output_img, equ)

    def gamma_correction(self, _path: str, _gamma: float):
        """
        ガンマ補正を行う機能。
        :param _path: 補正する画像のパス。
        :param _gamma: ガンマ値。
        :return:
        """
        img = cv2.imread(_path, 0)
        img_gamma = np.zeros_like(img, dtype='uint16')
        m = img.max()
        for h in tqdm(range(img.shape[0])):
            for w in range(img.shape[1]):
                img_gamma[h][w] = m * pow(img[h][w] / m, 1 / _gamma)
        # res = np.hstack((img, img_gamma))
        # cv2.imshow('corrected.', res)
        # cv2.waitKey(0)
        output_img = ''.join(_path.split('/')[-1].split('.')[0:-1]) + '_' + str(_gamma) + '.png'
        cv2.imwrite(output_img, img_gamma)

    def extract_img(self, _h: int, _w: int, _path: str):
        """
        画像を定形に細切れにする機能。
        :param _h: 細切れにする高さ。
        :param _w: 細切れにする幅。
        :param _path: 細切れにする画像のパス。
        :return:
        """
        img = cv2.imread(_path, 0)
        if img.shape[0] % _h != 0 or img.shape[1] % _w != 0:
            print('[Warn] Image cannot be divided by _w, or _h.')
            exit(1)
        for ew in range(0, len(img[1]), _w):
            for eh in range(0, len(img[0]), _h):
                extracted_patch = img[eh:eh + _h - 1, ew:ew + _w - 1]
                cv2.imwrite(
                    './' + _path.split('/')[-1].split('.')[0] + '_extpatch_h' + str(eh) + '_w' + str(ew) + '.png',
                    extracted_patch)