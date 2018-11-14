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
        img = cv2.imread(_path, 2)
        hist, bins = np.histogram(img.flatten(), img.max() + 1, [0, img.max()])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * img.max() / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint16')

        img2 = cdf[img]
        output_fol = '/'.join(_path.split('/')[0:-1]) + '/smoothed'
        print(output_fol)
        if not os.path.isdir(output_fol):
            os.mkdir(output_fol)
        output_img = ''.join(_path.split('/')[-1].split('.')[0:-1]) + '_smoothed.png'
        print(output_fol + '/' + output_img)
        cv2.imwrite(output_fol + '/' + output_img, img2)

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

    def extract_imgs(self, _h: int, _w: int, _path: str):
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

    def extract_img(self, _s: list, _h: int, _w: int, _path: str):
        """
        画像から単一の画像を切り取る処理
        :param _s: 切り取る開始位置（リスト型変数で、[x,y]）
        :param _h: 切り取る画像の高さ
        :param _w: 切り取る画像の幅
        :param _path: 切り取る基の画像のパス
        :return:
        """
        if len(_s) != 2:
            print('_s must be list as [x, y].')
            exit(1)
        img = cv2.imread(_path, 0)
        ext_img = img[_s[0]:_s[0] + _h, _s[1]:_s[1] + _w]
        output_fol = './test'
        if not os.path.isdir(output_fol):
            os.mkdir(output_fol)
        cv2.imwrite(output_fol + _path.split('/')[-1].split('.')[0] + '_ext_h' + str(_h) + '_w' + str(_w) + '.png',
                    ext_img)

    def extract_random_img(self, _num: int, _h: int, _w: int, _path: str):
        """
        特定の画像から、決まったサイズの画像をランダムに切り取る処理
        :param _num: 切り取る画像の数
        :param _h: 切り取る画像の高さ
        :param _w: 切り取る画像の幅
        :param _path: 切り取る画像のパス
        :return:
        """
        import random
        img = cv2.imread(_path, 0)
        for count in range(0, _num):
            width = random.randint(0, len(img[0, :]) - _w)
            height = random.randint(0, len(img[:, 0]) - _h)
            ext_img = img[height:height + _h, width:width + _w]
            cv2.imwrite('./preprocessing/ext_random/' + str(count) + '.png', ext_img)
