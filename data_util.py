from __future__ import print_function
import glob
import numpy as np
import os
import sys
import urllib
import zipfile


def _read_data(signal_files):
    """
    load sensor data into a tensor 

    :param signal_files: a list of sensor data files
    :return: a tensor represents sensor data
    """
    signal_tensor = np.dstack([np.loadtxt(signal_file) for signal_file in signal_files])
    if not __debug__:
        print('loaded input tensor shape: ', signal_tensor.shape)
    return signal_tensor


def _read_label(label_file):
    """
    load label data into a one-hot tensor

    :param label_file: the file that contains the label
    :return: a one-hot represented tensor
    """
    y_ = np.loadtxt(label_file, dtype=np.int32)

    if not __debug__:
        print('loaded label shape: ', y_.shape)

    # use one-hot encoding
    y_one_hot = np.zeros((y_.size, y_.max()))
    y_one_hot[np.arange(y_.size), y_ - 1] = 1

    return y_one_hot


def _list_files(path):
    """
    get the files under a folder
    :param path: data path
    :return: a list of data files
    """
    return [os.path.join(path, data_file) for data_file in os.listdir(path)
            if os.path.isfile(os.path.join(path, data_file))]


def get_data(data_type=None):
    if data_type is None or not (data_type.lower() == "train" or data_type.lower() == "test"):
        raise TypeError("data_type must be either train or test, you gave: {}".format(data_type))

    data_path = "data/UCI HAR Dataset/"
    signal_path = "Inertial Signals/"
    type_path = "{}/".format(data_type)
    label_file = "y_{}.txt".format(data_type)

    input_data_path = data_path + type_path + signal_path + '/*.txt'
    input_data_files = sorted(glob.glob(input_data_path))

    if not __debug__:
        print(input_data_files)

    x = _read_data(input_data_files)

    label_file = data_path + type_path + label_file
    y = _read_label(label_file)

    return x, y


def _dl_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r%2d%%" % percent)
    sys.stdout.flush()


def maybe_prepare_data():
    print("")
    extract_directory = os.path.abspath("data/UCI HAR Dataset")
    data_file = "data/UCI HAR Dataset.zip"

    if os.path.exists(extract_directory):
        print("UCI HAR Dataset already prepared")
        print("")
        return
    else:
        if os.path.exists(data_file):
            print("UCI HAR Dataset.zip already downloaded")
        else:
            print("Downloading UCI HAR Dataset...")
            if not os.path.exists("data"):
                os.mkdir("data")
            urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.zip",
                               data_file, reporthook=_dl_progress)
            print("UCI HAR Dataset.zip downloaded")
        print("Extracting UCI HAR Dataset.zip...")
        with zipfile.ZipFile(data_file, "r") as zip_ref:
            zip_ref.extractall("data")
            print("UCI HAR Dataset prepared")
        print("")


def zip_files(zip_file, files):
    """
    :param zip_file: zip file name
    :param files: pattern of a list of files
    :return: none
    """
    file_path = os.path.dirname(zip_file)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    with zipfile.ZipFile(zip_file, "w") as zip_ref:
        for file_item in glob.glob(files):
            print("adding {} to {}".format(file_item, zip_file))
            zip_ref.write(file_item)

# if __name__ == '__main__':
#     for model_name in ["2layer32unit", "2layer64unit", "2layer128unit", "2layer256unit", "3layer64unit"]:
#         zip_files("model/{}.ckpt.zip".format(model_name), "data/{}.ckpt.*".format(model_name))
#         zip_files("model/{}.model.zip".format(model_name), "data/{}.pb*".format(model_name))
