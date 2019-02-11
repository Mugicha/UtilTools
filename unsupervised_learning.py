from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np


class util:
    def __init__(self):
        pass

    def silhouette_coefficient(self, _x, _y):
        """
        シルエット関数を使ってクラスタリングの結果を検証する機能
        :param _x: 学習データ
        :param _y: クラスタリング結果
        :return:
        """
        cluster_labels = np.unique(_y)
        n_clusters = cluster_labels.shape[0]

        # calc silhoette coefficient
        silhouette_vals = silhouette_samples(_x, _y, metric='euclidean')
        y_as_lower, y_as_upper = 0, 0
        yticks = []
        for i, c in enumerate(cluster_labels):
            c_silhouette_vals = silhouette_vals[_y == c]
            c_silhouette_vals.sort()
            y_as_upper += len(c_silhouette_vals)
            color = cm.jet(float(i) / n_clusters)
            plt.barh(range(y_as_lower, y_as_upper),
                     c_silhouette_vals, height=1.0, edgecolor='none', color=color)
            yticks.append((y_as_lower + y_as_upper) / 2)
            y_as_lower += len(c_silhouette_vals)
        silhouette_average = np.mean(silhouette_vals)
        plt.axvline(silhouette_average, color='red', linestyle='--')
        plt.yticks(yticks, cluster_labels + 1)
        plt.ylabel('Cluster')
        plt.xlabel('Silhouette coefficient')
        plt.tight_layout()
        plt.show()


class Kmeans_Clustering:
    """
    K-meansを扱うクラス
    """
    def __init__(self):
        pass

    def train(self, _n_class: int, _x, _concat: bool = False):
        """
        k-meansでクラスタリングを行う
        :param _n_class: 分割したいクラス数
        :param _x: 学習で使用するデータ
        :param _concat: 学習データと分割データを結合して返すかどうか（default: False）
        :return:
        """
        km = KMeans(n_clusters=_n_class, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
        y_km = km.fit_predict(_x)  # np.ndarray
        if _concat:
            return np.hstack([_x.values, y_km.reshape(len(y_km), 1)])
        else:
            return y_km

    def elbow(self, _max_cluster: int, _x):
        """
        エルボー法で、最適なクラスタ数を求める機能。
        :param _max_cluster: 検証する分割クラスタの最大数
        :param _x: 学習用データ
        :return:
        """
        distortions = []
        for i in range(1, _max_cluster + 1):
            km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
            km.fit(_x)
            distortions.append(km.inertia_)
        plt.title('elbow method result')
        plt.xlabel('num of cluster')
        plt.ylabel('distortions')
        plt.plot(range(1, _max_cluster + 1), distortions)
        plt.show()


class Agglomerative_Clustering:
    def __init__(self):
        pass

    def train(self, n_class: int, _x, _concat: bool = False):
        """
        凝集型階層的クラスタリングを行う機能
        :param n_class: 分割したいクラス数
        :param _x: 学習で使用するデータ
        :param _concat: 学習データと分割データを結合して返すかどうか（default: False）
        :return:
        """
        ac = AgglomerativeClustering(n_clusters=n_class, affinity='euclidean', linkage='complete')
        y_ac = ac.fit_predict(_x)
        if _concat:
            return np.hstack([_x.values, y_ac.reshape(len(y_ac), 1)])
        else:
            return y_ac


class Dbscan:
    def __init__(self):
        pass

    def train(self, _x, _eps: float = 0.2, _min_samples: int = 5, _concat: bool = False):
        """
        DBSCANのクラスタリングを行う機能
        :param _x: 学習で使用するデータ
        :param _eps: 隣接点とみなす2点間の最大距離
        :param _min_samples: ボーダー点の最小個数
        :param _concat: 学習データと分割データを結合して返すかどうか（default: False）
        :return:
        """
        db = DBSCAN(eps=_eps, min_samples=_min_samples, metric='euclidean')
        y_db = db.fit_predict(_x)
        if _concat:
            return np.hstack([_x.values, y_db.reshape(len(y_db), 1)])
        else:
            return y_db
