
import pickle
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import keras_cv.models
from keras_cv import bounding_box
from keras_cv import visualization

import generate_synthetic_data as gsd
import script_utility as su


class EvaluateCOCOMetricsCallback(keras.callbacks.Callback):
    """
    ## COCO Metric Callback

    We will be using `BoxCOCOMetrics` from KerasCV to evaluate the model and calculate the
    Map(Mean Average Precision) score, Recall and Precision. We also save our model when the
    mAP score improves.
    """

    def __init__(self, data, save_path):
        super().__init__()
        self.data = data
        self.metrics = keras_cv.metrics.BoxCOCOMetrics(
            bounding_box_format="xyxy",
            evaluate_freq=1e9,
        )

        self.save_path = save_path
        self.best_map = -1.0

    def on_epoch_end(self, epoch, logs):
        self.metrics.reset_state()
        for batch in self.data:
            images, y_true = batch[0], batch[1]
            y_pred = self.model.predict(images, verbose=0)
            self.metrics.update_state(y_true, y_pred)

        metrics = self.metrics.result(force=True)
        logs.update(metrics)

        current_map = metrics["MaP"]
        if current_map > self.best_map:
            self.best_map = current_map
            self.model.save(self.save_path)  # Save the model when mAP improves

        return logs


def minimal_inference_examples():
    input_data = tf.ones(shape=(8, 224, 224, 3))
    #  Pretrained backbone
    model = keras_cv.models.YOLOV8Backbone.from_preset(
        "yolo_v8_xs_backbone_coco"
    )
    output = model(input_data)
    print(f"{output=}")

    # Randomly initialized backbone with a custom config
    model = keras_cv.models.YOLOV8Backbone(
        stackwise_channels=[128, 256, 512, 1024],
        stackwise_depth=[3, 9, 9, 3],
        include_rescaling=False,
    )
    output = model(input_data)
    print(f"{output=}")

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


def load_dataset(image_path, classes, bbox):
    # Read Image
    image = load_image(image_path)
    bounding_boxes = {
        "classes": tf.cast(classes, dtype=tf.float32),
        "boxes": bbox,
    }
    return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}


def visualize_dataset(inputs, value_range, rows, cols, bounding_box_format, class_mapping):
    inputs = next(iter(inputs.take(1)))
    images, bounding_boxes = inputs["images"], inputs["bounding_boxes"]
    visualization.plot_bounding_box_gallery(
        images,
        value_range=value_range,
        rows=rows,
        cols=cols,
        y_true=bounding_boxes,
        scale=5,
        font_scale=0.7,
        bounding_box_format=bounding_box_format,
        class_mapping=class_mapping,
    )


def train_yolo(data_lst: List[Tuple[Path, List, List]]):
    SPLIT_RATIO = 0.2
    BATCH_SIZE = 4
    LEARNING_RATE = 0.001
    EPOCH = 5
    GLOBAL_CLIPNORM = 10.0

    class_ids = ["square", "circle"]
    class_mapping = dict(zip(range(len(class_ids)), class_ids))


    image_paths = [str(x[0]) for x in data_lst]
    classes = [x[1] for x in data_lst]
    bbox = [x[2] for x in data_lst]

    bbox = tf.ragged.constant(bbox)
    classes = tf.ragged.constant(classes)
    image_paths = tf.ragged.constant(image_paths)

    data = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))

    # Determine the number of validation samples
    num_val = int(len(data_lst) * SPLIT_RATIO)

    # Split the dataset into train and validation sets
    val_data = data.take(num_val)
    train_data = data.skip(num_val)

    augmenter = keras.Sequential(
        layers=[
            keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xyxy"),
            keras_cv.layers.RandomShear(
                x_factor=0.2, y_factor=0.2, bounding_box_format="xyxy"
            ),
            keras_cv.layers.JitteredResize(
                target_size=(640, 640), scale_factor=(0.75, 1.3), bounding_box_format="xyxy"
            ),
        ]
    )

    """
    ## Creating Training Dataset
    """

    train_ds = train_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(BATCH_SIZE * 4)
    train_ds = train_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
    train_ds = train_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)

    """
    ## Creating Validation Dataset
    """

    resizing = keras_cv.layers.JitteredResize(
        target_size=(640, 640),
        scale_factor=(0.75, 1.3),
        bounding_box_format="xyxy",
    )

    val_ds = val_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.shuffle(BATCH_SIZE * 4)
    val_ds = val_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
    # val_ds = val_ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)

    backbone = keras_cv.models.YOLOV8Backbone.from_preset(
        "yolo_v8_s_backbone_coco"  # We will use yolov8 small backbone with coco weights
    )

    yolo = keras_cv.models.YOLOV8Detector(
        num_classes=2,
        bounding_box_format="xyxy",
        backbone=backbone,
        fpn_depth=1,
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        global_clipnorm=GLOBAL_CLIPNORM,
    )

    yolo.compile(optimizer=optimizer, classification_loss="binary_crossentropy", box_loss="ciou")

    yolo.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3,
        callbacks=[EvaluateCOCOMetricsCallback(val_ds, "model.h5")],
    )


def maybe_create_data() -> List[Tuple[Path, List, List]]:
    out_folder = Path.home() / 'minimal_yolo_synth_data'
    if not out_folder.exists():
        out_folder.mkdir()

    pickle_path = out_folder / gsd.PICKLE_FILE_NAME

    if not pickle_path.exists():
        data = gsd.generate_synthetic_data_and_save(out_folder, 10000, 224, 224, 3,
                                                       (10, 40))
    else:
        with open(out_folder / gsd.PICKLE_FILE_NAME, 'rb') as f:
            data = pickle.load(f)

    return data


@su.print_runtime
def main():
    su.set_logging()
    data = maybe_create_data()
    train_yolo(data)


if __name__ == "__main__":
    main()
