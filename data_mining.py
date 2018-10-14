from sklearn.decomposition import PCA
import pandas as pd


class DataManipulation():

    def pca_reduction(self, _df: pd.DataFrame, _dim: int, _return_with_model=False):
        pca = PCA(n_components=_dim)  # type: PCA
        pca.fit(_df.values)
        transformed = pca.fit_transform(_df.values)  # TODO write return type.
        if _return_with_model:
            return transformed, pca
        else:
            return transformed

