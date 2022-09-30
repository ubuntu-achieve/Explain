import numpy as np
from keras.datasets import mnist

NO_OF_POINTS = 250
NO_BASELINES = 3
BASELINES_ZERO = "zero"
BASELINES_ENCODER_ZERO = "encoded_zero"
BASELINES_MEDIAN = "median"
BASELINES_MEAN = "mean"
MAX_EPOCHS = 100
DISPLAY_STEPS = 1
EARLY_STOPPING = 10
IMAGE_SIZE = 28
VALIDATION_SIZE = 5000
NUM_CHANNELS = 1
PIXEL_DEPTH = 255.
NUM_LABELS = 10
BATCH_SIZE = 128
MAX_SAMPLES = 300
MAX_POINTS_FOR_MEAN = 20000
MAX_NEIGHBORS = 1000


def read_input_data_cnn():
    """
    load the mnist data for a CNN network.
    :return: Dict, mnist data and labels
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Get one hot encoding for train and test labels
    one_hot_encoding = np.zeros((len(y_train), NUM_LABELS))
    one_hot_encoding[np.arange(len(y_train)), y_train] = 1
    one_hot_encoding = np.reshape(one_hot_encoding, [-1, NUM_LABELS])
    y_train = one_hot_encoding

    one_hot_encoding = np.zeros((len(y_test), NUM_LABELS))
    one_hot_encoding[np.arange(len(y_test)), y_test] = 1
    one_hot_encoding = np.reshape(one_hot_encoding, [-1, NUM_LABELS])
    y_test = one_hot_encoding

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

    # normalize to float values on [0,1]
    x_train = x_train / PIXEL_DEPTH
    x_test = x_test / PIXEL_DEPTH

    data = dict()
    data["x"] = np.concatenate((x_train, x_test))
    data["y"] = np.concatenate((y_train, y_test))
    print("Data shape: {}, Data labels: {}".format(data["x"].shape, data["y"].shape))
    return data


def get_samples(data_dict,
                sample_type,
                num_samples):
    """
    Get the example_digits asked for by sample type
    :param data_dict: dict, dictionary containing example_digits images and labels
    :param sample_type: str, example_digits needed. e.g. "5"
    :param num_samples: int, number of events needed
    :return: images for the sample_type
    """
    x = data_dict["x"]
    y = data_dict["y"]
    sample_digit = int(sample_type)
    sample_idx = np.where(y[:, sample_digit] == 1)[0]
    x_digit = x[sample_idx]
    if x_digit.shape[0] >= num_samples:
        x_digit = x_digit[0:num_samples]
    print("Digit: {}, Data size: {}".format(sample_type, x_digit.shape))
    return x_digit


def get_neighbors_data(data_dict):
    """
    Get neighbors data and flatten data to two dimensions.
    :param data_dict: Dict, mnist data and labels
    :return: np.array, neighbors data
    """
    x = data_dict["x"]
    y = data_dict["y"]
    x_neighbors = []
    for i in range(NUM_LABELS):
        sample_idx = np.where(y[:, i] == 1)[0]
        if len(sample_idx) > MAX_NEIGHBORS:
            sample_idx = sample_idx[0:MAX_NEIGHBORS]
        x_neighbors.extend(list(x[sample_idx]))
    x_neighbors = np.array(x_neighbors)
    x_neighbors_flatten = []
    for ii in x_neighbors:
        x_neighbors_flatten.append(ii.flatten())

    x_neighbors_flatten = np.array(x_neighbors_flatten)
    print("neighbors data shape: ", x_neighbors_flatten.shape)
    return x_neighbors_flatten


def get_neighbors_data_cnn(data_dict):
    """
    Get neighbors data without flattening it.
    :param data_dict: Dict, mnist data and labels
    :return: np.array, neighbors data
    """
    x = data_dict["x"]
    y = data_dict["y"]
    x_neighbors = []
    for i in range(NUM_LABELS):
        sample_idx = np.where(y[:, i] == 1)[0]
        if len(sample_idx) > MAX_NEIGHBORS:
            sample_idx = sample_idx[0:MAX_NEIGHBORS]
        x_neighbors.extend(list(x[sample_idx]))
    x_neighbors = np.array(x_neighbors)
    print("neighbors data shape: ", x_neighbors.shape)
    return x_neighbors

