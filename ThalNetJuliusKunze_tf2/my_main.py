import itertools
from pathlib import Path

import numpy as np
import tensorflow as tf

from model import SequenceClassifier, ThalNetCell, GRUCell,MLPClassifier
from util import timestamp
import argparse
from plot import plot_learning_curve

def plot_image(image: np.ndarray, label: str) -> None:
    from matplotlib import pyplot as plt
    plt.title(f'Label {label}')
    plt.imshow(image)
    plt.show()

(xs, ys), (_, _) = tf.keras.datasets.mnist.load_data()

num_classes = 10
num_rows, row_size = 28, 28

# summary_writer = tf.compat.v1.summary.FileWriter(str(log_path), graph=tf.compat.v1.get_default_graph())

(xs, ys), (_, _) = tf.keras.datasets.mnist.load_data()

xs = (xs / 255.).astype('float32')

get_thalnet_cell = lambda: ThalNetCell(input_size=row_size, output_size=num_classes, context_input_size=32,
                                       center_size_per_module=32, num_modules=4)

data = xs[0, :, :]
target = ys[0]
dropout = 0

model = SequenceClassifier(data, target, dropout, get_rnn_cell=get_thalnet_cell, num_rows=num_rows, row_size=row_size)

