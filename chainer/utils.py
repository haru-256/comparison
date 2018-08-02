from tqdm import tqdm
import datetime
import copy
import pathlib
# import matplotlib as mpl
# mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shutil
from sklearn import model_selection
import pandas


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
    val_dir = (data_dir / "val").resolve()
    if not val_dir.exists():
        val_dir.mkdir()
        print("make directory for validation dataset:", val_dir)
    train_data_dir = (data_dir / "train").resolve()
    paths = [[path, path.parts[-2]] for path in train_data_dir.glob("*/*.jpg")]
    df = pandas.DataFrame(paths, columns=["path", "class"])
    _, val_data_path = model_selection.train_test_split(df.loc[:, "path"], test_size=test_size,
                                                        stratify=df["class"], random_state=seed)
    for path in val_data_path:
        class_dir = val_dir / path.parts[-2]  # クラスラベルのディレクトリ
        if not class_dir.exists():
            class_dir.mkdir()

        # 画像ファイルの移動
        shutil.move(path, val_dir / "/".join(path.parts[-2:]))
    print("Done")

    return val_dir, val_data_path
