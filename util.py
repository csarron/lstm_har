from __future__ import print_function
import numpy as np
import os
import sys
import tensorflow as tf
import urllib
import zipfile


def read_data(signal_files):
    """
    load sensor data into a tensor 

    :param signal_files: a list of sensor data files
    :return: a tensor represents sensor data
    """
    signal_tensor = np.dstack([np.loadtxt(signal_file) for signal_file in signal_files])
    if not __debug__:
        print('loaded input tensor shape: ', signal_tensor.shape)
    return signal_tensor


def read_label(label_file):
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


def list_files(path):
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

    input_data_path = data_path + type_path + signal_path

    input_data_files = list_files(input_data_path)

    if not __debug__:
        print(input_data_files)

    x = read_data(input_data_files)

    label_file = data_path + type_path + label_file
    y = read_label(label_file)

    return x, y


def dl_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r%2d%%" % percent)
    sys.stdout.flush()


def maybe_download_data():
    print("")

    print("Downloading UCI HAR Dataset...")
    data_file = "data/UCI HAR Dataset.zip"
    if not os.path.exists(data_file):
        if not os.path.exists("data"):
            os.mkdir("data")
        urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.zip",
                           data_file, reporthook=dl_progress)
        print(" downloaded.")
    else:
        print("UCI HAR Dataset.zip already downloaded.")

    print("Extracting UCI HAR Dataset.zip...")
    extract_directory = os.path.abspath("data/UCI HAR Dataset")
    if not os.path.exists(extract_directory):
        with zipfile.ZipFile(data_file, "r") as zip_ref:
            zip_ref.extractall("data")
    else:
        print("Dataset already extracted.")


def get_tensor_by_op_name(g, names):
    return [g.get_operation_by_name(n).outputs[0] for n in names]


def get_tensor_by_name(g, names):
    return [g.get_tensor_by_name(n).outputs[0] for n in names]


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="",
            op_dict=None,
            producer_op_list=None
        )
    return graph
