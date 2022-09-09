"""
Title: Tune a RetinaNet Prediction Decoder
Author: [lukewood](https://lukewood.xyz)
Date created: 2022/09/08
Last modified: 2022/09/08
Description: Learn how to search for strong prediction decoder settings in KerasCV.
"""

"""
## Overview

In [the introduction to KerasCV RetinaNet tutorial](https://keras.io/guides/keras_cv/retina_net_overview/),
we cover how to train a KerasCV RetinaNet on the PascalVOC dataset.
This guide shows you how to take a pre-trained RetinaNet and tune the PredictionDecoder
layer to achieve an optimal Mean Average Precision (MaP).

To run this guide, you should first follow the [KerasCV RetinaNet tutorial](https://keras.io/guides/keras_cv/retina_net_overview/)
to produce a pretrained checkpoint.
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import optimizers

import keras_cv
from keras_cv import bounding_box
import os

BATCH_SIZE = 8
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "checkpoint/")

"""
"""
