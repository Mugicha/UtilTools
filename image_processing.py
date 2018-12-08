import cv2
import os
import numpy as np
from tqdm import tqdm
import UtilTools.common


class Image_processing():
    def __init__(self):
        self.c = UtilTools.common.Common()

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
        if self.c.folder_check(output_fol):
            output_img = ''.join(_path.split('/')[-1].split('.')[0:-1]) + '_smoothed.png'
            cv2.imwrite(os.path.join(output_fol, '/' + output_img), img2)

    def gamma_correction(self, _path: str, _gamma: float, _outpath='./'):
        """
        ガンマ補正を行う機能。
        :param _path: 補正する画像のパス。
        :param _gamma: ガンマ値。
        :param _outpath: 出力先のフォルダ
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
        if self.c.folder_check(_outpath):
            cv2.imwrite(os.path.join(_outpath, output_img), img_gamma)

    def extract_imgs(self, _h: int, _w: int, _path: str, _outpath='./'):
        """
        画像を定形に細切れにする機能。
        :param _h: 細切れにする高さ。
        :param _w: 細切れにする幅。
        :param _path: 細切れにする画像のパス。
        :param _outpath: 細切れにした画像を保存するフォルダのパス。
        :return:
        """
        img = cv2.imread(_path, 0)
        if img.shape[0] % _h != 0 or img.shape[1] % _w != 0:
            print('[Warn] Image cannot be divided by _w, or _h.')
            exit(1)
        if not self.c.folder_check(_outpath):
            exit(1)
        for ew in range(0, len(img[1]), _w):
            for eh in range(0, len(img[0]), _h):
                extracted_patch = img[eh:eh + _h - 1, ew:ew + _w - 1]
                cv2.imwrite(os.path.join(_outpath, _path.split('/')[-1].split('.')[0] + '_extpatch_h' + str(eh) + '_w' + str(ew) + '.png'), extracted_patch)

    def extract_img(self, _s: list, _h: int, _w: int, _path: str, _outpath='./'):
        """
        画像から単一の画像を切り取る処理
        :param _s: 切り取る開始位置（リスト型変数で、[x,y]）
        :param _h: 切り取る画像の高さ
        :param _w: 切り取る画像の幅
        :param _path: 切り取る基の画像のパス
        :param _outpath: 出力先のフォルダのパス
        :return:
        """
        if len(_s) != 2:
            print('_s must be list as [x, y].')
            exit(1)
        img = cv2.imread(_path, 0)
        ext_img = img[_s[0]:_s[0] + _h, _s[1]:_s[1] + _w]
        if self.c.folder_check(_outpath):
            cv2.imwrite(os.path.join(_outpath, _path.split('/')[-1].split('.')[0] + '_ext_h' + str(_h) + '_w' + str(_w) + '.png'), ext_img)

    def extract_random_img(self, _num: int, _h: int, _w: int, _path: str, _outpath='./'):
        """
        特定の画像から、決まったサイズの画像をランダムに切り取る処理
        :param _num: 切り取る画像の数
        :param _h: 切り取る画像の高さ
        :param _w: 切り取る画像の幅
        :param _path: 切り取る画像ファイルのパス
        :param _outpath: 出力先のフォルダのパス
        :return:
        """
        import random
        img = cv2.imread(_path, 0)
        for count in range(0, _num):
            width = random.randint(0, len(img[0, :]) - _w)
            height = random.randint(0, len(img[:, 0]) - _h)
            ext_img = img[height:height + _h, width:width + _w]
            if self.c.folder_check(_outpath):
                cv2.imwrite(os.path.join(_outpath, str(count) + '.png'), ext_img)