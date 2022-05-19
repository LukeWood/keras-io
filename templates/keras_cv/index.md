# KerasCV

KerasCV is a toolbox of modular building blocks (layers, metrics, losses, data augmentation) that computer vision engineers can leverage to quickly assemble production-grade, state-of-the-art training and inference pipelines for common use cases such as image classification, object detection, image segmentation, image data augmentation, etc.

---
## Quick links

* [KerasCV developer guides](/guides/keras_cv/)
* [KerasCV API reference](/api/keras_cv/)
* [KerasCV on GitHub](https://github.com/keras-team/keras-cv)

---
## Installation

KerasCV requires **Python 3.7+** and **TensorFlow 2.9+**.

Install the latest release:

```
pip install keras-cv --upgrade
```

You can also check out other versions in our
[GitHub repository](https://github.com/keras-team/keras-cv/releases).

## Quickstart
The following snippet will allow you to create a simple ResNet18 and train it with
data augmented by MixUp, CutMix, and RandAugment.

```python
import tensorflow as tf
import keras_cv

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

preprocessing_model = keras.Sequential(
    [
        keras_cv.layers.MixUp(),
        keras_cv.layers.CutMix(),
        keras_cv.layers.RandAugment(magnitude=0.2),
    ]
)
model = keras_cv.applications.ResNet18V2(
    include_preprocessing=True, num_classes=10, weights=None
)

train_dataset = tf.data.Dataset.from_tensor_slices(
  {"images": x_train, "targets": y_train}
)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_dataset = train_dataset.shuffle(BATCH_SIZE * 3).batch(BATCH_SIZE)
train_dataset = train_dataset.map(
    preprocessing_model, num_parallel_calls=tf.data.AUTOTUNE
)
train_dataset = train_dataset.map(
  lambda inputs: (inputs["images"], inputs["targets"]),
  num_parallel_calls=tf.data.AUTOTUNE
)

model.compile(
  loss="sparse_categorical_crossentropy",
  optimizer="adam",
  metrics=["accuracy"]
)

model.fit(train_dataset, validation_data=test_dataset, epochs=10)
```

For an end-to-end example using KerasCV check out our __TODO__ guide.

---
## Citing KerasCV

If KerasCV helps your research, we appreciate your citations.
Here is the BibTeX entry:

```bibtex
@misc{wood2022kerascv,
  title={KerasCV},
  author={Wood, Luke and Zhu, Scott and Chollet, Fran\c{c}ois and others},
  year={2022},
  howpublished={\url{https://github.com/keras-team/keras-cv}},
}
```
