from tqdm import tqdm
import datetime
import copy
import pathlib
# import matplotlib as mpl
# mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shutil
from sklearn import model_selection, preprocessing
import pandas as pd
import chainercv
from chainercv import transforms
from chainer import dataset, training


class PytorchLike_LabeledImageDataset(dataset.DatasetMixin):
    """
    datasets class that assumes file structure like Pytorch.

    Parameters
    -------------------------
    dir_path: pathlib.Path
        path to directory to hold the data.

    transform_list: list
        list to holds transformars that collable function.

    param_list: list of dictionary
        lisy of dictionary to pass to transform.
    """

    def __init__(self, dir_path, transform_list, param_list):
        self.transform_list = transform_list
        self.param_list = param_list
        if not dir_path.is_absolute():
            dir_path.resolve()
        # label
        labels = [path.parts[-1] for path in dir_path.glob('*')]
        self.le = preprocessing.LabelEncoder().fit(labels)

        # store paths and labels into list
        self._pairs = pd.DataFrame(
            [(path, path.parts[-2]) for path in dir_path.glob('*/*.jpg')], columns=["path", "label"])
        self._pairs.label = self.le.transform(self._pairs.label)

    def __len__(self):
        return len(self._pairs)

    def get_example(self, i):
        path, label = self._pairs.iloc[i]
        # return images whise datafoirmat is CHW
        image = chainercv.utils.read_image(path)
        for param, transform in zip(self.param_list, self.transform_list):
            image = transform(image, **param)
        label = np.array(label, dtype=np.int32)

        return image, label


class NormUpdater(training.updaters.StandardUpdater):
    def __init__(self, iterator, optimizer, normalize_param, converter=dataset.convert.concat_examples,
                 device=None, loss_func=None, loss_scale=None):
        self.normalize_param = normalize_param
        super.__init__(iterator, optimizer, converter=dataset.convert.concat_examples,
                       device=None, loss_func=None, loss_scale=None)

    def update_core(self):
        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)

        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target

        if isinstance(in_arrays, tuple):
            optimizer.update(loss_func, *in_arrays)
        elif isinstance(in_arrays, dict):
            optimizer.update(loss_func, **in_arrays)
        else:
            optimizer.update(loss_func, in_arrays


def make_validation_dataset(data_dir, seed=None, test_size=0.25):
    """
    make validation dataset using immigrating data

    Parameters
    ---------------------
    data_dir: pathlib.Path
        path to data directory

    seed: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.

    test_size : float, int, None, optional
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset
        to include in the test split. If int, represents the absolute number of test samples.
        If None, the value is set to the complement of the train size. By default,
        the value is set to 0.25. The default will change in version 0.21. It will remain 0.25
        only if train_size is unspecified, otherwise it will complement the specified train_size.

    Returns
    ------------------------
    val_dir: pathlib.Path
        path to validation datasets directory

    val_data_path: pandas.Series
        pandas.Series object whose each elements is path to validation data
    """
    # data immigration for validation data
    print("make validation dataset")
    val_dir=(data_dir / "val").resolve()
    if not val_dir.exists():
        val_dir.mkdir()
        print("make directory for validation dataset:", val_dir)
    train_data_dir=(data_dir / "train").resolve()
    paths=[[path, path.parts[-2]] for path in train_data_dir.glob("*/*.jpg")]
    df=pd.DataFrame(paths, columns=["path", "class"])
    _, val_data_path=model_selection.train_test_split(df.loc[:, "path"], test_size=test_size,
                                                        stratify=df["class"], random_state=seed)
    for path in val_data_path:
        class_dir=val_dir / path.parts[-2]  # クラスラベルのディレクトリ
        if not class_dir.exists():
            class_dir.mkdir()

        # 画像ファイルの移動
        shutil.move(path, val_dir / "/".join(path.parts[-2:]))
    print("Done")

    return val_dir, val_data_path
