
from pathlib import Path

import cv2
import h5py
import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from IPython.display import Image
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Activation,
    Conv3D,
    Conv3DTranspose,
    MaxPooling3D,
    UpSampling3D,
    concatenate,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

K.set_image_data_format("channels_first")


# =========================
# Visualization
# =========================
def plot_image_grid(image):
    data_all = [image]

    fig, ax = plt.subplots(3, 6, figsize=(16, 9))

    coronal = np.transpose(data_all, [1, 3, 2, 4, 0])
    coronal = np.rot90(coronal, 1)

    transversal = np.transpose(data_all, [2, 1, 3, 4, 0])
    transversal = np.rot90(transversal, 2)

    sagittal = np.transpose(data_all, [2, 3, 1, 4, 0])
    sagittal = np.rot90(sagittal, 1)

    planes = [
        (coronal, "Coronal"),
        (transversal, "Transversal"),
        (sagittal, "Sagittal"),
    ]

    for row, (plane, name) in enumerate(planes):
        for i in range(6):
            n = np.random.randint(plane.shape[2])
            ax[row, i].imshow(
                np.squeeze(plane[:, :, n, :])
            )
            ax[row, i].set_xticks([])
            ax[row, i].set_yticks([])

            if i == 0:
                ax[row, i].set_ylabel(
                    name, fontsize=15
                )

    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def visualize_data_gif(data_):
    images = []

    for i in range(data_.shape[0]):
        x = data_[
            min(i, data_.shape[0] - 1), :, :
        ]
        y = data_[
            :, min(i, data_.shape[1] - 1), :
        ]
        z = data_[
            :, :, min(i, data_.shape[2] - 1)
        ]

        img = np.concatenate(
            (x, y, z), axis=1
        )
        images.append(img)

    imageio.mimsave(
        "gif.gif", images, duration=0.01
    )

    return Image(filename="gif.gif")


# =========================
# Convolution blocks
# =========================
def create_convolution_block(
    input_layer,
    n_filters,
    batch_normalization=False,
    kernel=(3, 3, 3),
    activation=None,
    padding="same",
    strides=(1, 1, 1),
    instance_normalization=False,
):
    layer = Conv3D(
        filters=n_filters,
        kernel_size=kernel,
        padding=padding,
        strides=strides,
    )(input_layer)

    if activation is None:
        return Activation("relu")(layer)

    return activation()(layer)


def get_up_convolution(
    n_filters,
    pool_size,
    kernel_size=(2, 2, 2),
    strides=(2, 2, 2),
    deconvolution=False,
):
    if deconvolution:
        return Conv3DTranspose(
            filters=n_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
        )

    return UpSampling3D(size=pool_size)


# =========================
# 3D U-Net architecture
# =========================
def unet_model_3d(
    loss_function,
    input_shape=(4, 160, 160, 16),
    pool_size=(2, 2, 2),
    n_labels=3,
    initial_learning_rate=0.00001,
    deconvolution=False,
    depth=4,
    n_base_filters=32,
    include_label_wise_dice_coefficients=False,
    metrics=None,
    batch_normalization=False,
    activation_name="sigmoid",
):
    inputs = Input(shape=input_shape)
    current_layer = inputs
    levels = []

    # Encoder
    for layer_depth in range(depth):
        filters = n_base_filters * (
            2 ** layer_depth
        )

        layer1 = create_convolution_block(
            input_layer=current_layer,
            n_filters=filters,
            batch_normalization=batch_normalization,
        )

        layer2 = create_convolution_block(
            input_layer=layer1,
            n_filters=filters * 2,
            batch_normalization=batch_normalization,
        )

        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(
                pool_size=pool_size
            )(layer2)

            levels.append(
                [layer1, layer2, current_layer]
            )
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # Decoder
    for layer_depth in range(
        depth - 2, -1, -1
    ):
        current_filters = K.int_shape(
            current_layer
        )[1]

        up_convolution = get_up_convolution(
            pool_size=pool_size,
            deconvolution=deconvolution,
            n_filters=current_filters,
        )(current_layer)

        concat = concatenate(
            [
                up_convolution,
                levels[layer_depth][1],
            ],
            axis=1,
        )

        skip_filters = K.int_shape(
            levels[layer_depth][1]
        )[1]

        current_layer = create_convolution_block(
            n_filters=skip_filters,
            input_layer=concat,
            batch_normalization=batch_normalization,
        )

        current_layer = create_convolution_block(
            n_filters=skip_filters,
            input_layer=current_layer,
            batch_normalization=batch_normalization,
        )

    final_convolution = Conv3D(
        n_labels,
        (1, 1, 1),
    )(current_layer)

    act = Activation(
        activation_name
    )(final_convolution)

    model = Model(
        inputs=inputs,
        outputs=act,
    )

    if metrics is None:
        metrics = []
    elif not isinstance(metrics, list):
        metrics = [metrics]

    model.compile(
        optimizer=Adam(
            learning_rate=initial_learning_rate
        ),
        loss=loss_function,
        metrics=metrics,
    )

    return model


# =========================
# Patch visualization
# =========================
def visualize_patch(X, y):
    fig, ax = plt.subplots(
        1, 2,
        figsize=(10, 5),
        squeeze=False,
    )

    ax[0][0].imshow(
        X[:, :, 0],
        cmap="Greys_r",
    )
    ax[0][0].set_yticks([])
    ax[0][0].set_xticks([])

    ax[0][1].imshow(
        y[:, :, 0],
        cmap="Greys_r",
    )
    ax[0][1].set_yticks([])
    ax[0][1].set_xticks([])

    fig.subplots_adjust(
        wspace=0, hspace=0
    )

    return fig


# =========================
# Data generator
# =========================
class VolumeDataGenerator(
    keras.utils.Sequence
):
    def __init__(
        self,
        sample_list,
        base_dir,
        batch_size=1,
        shuffle=True,
        dim=(160, 160, 16),
        num_channels=4,
        num_classes=3,
        verbose=1,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.base_dir = Path(base_dir)
        self.dim = dim
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.verbose = verbose
        self.sample_list = sample_list

        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(
            len(self.sample_list)
        )

        if self.shuffle:
            np.random.shuffle(
                self.indexes
            )

    def __len__(self):
        return int(
            np.floor(
                len(self.sample_list)
                / self.batch_size
            )
        )

    def __data_generation(
        self, list_IDs_temp
    ):
        X = np.zeros(
            (
                self.batch_size,
                self.num_channels,
                *self.dim,
            ),
            dtype=np.float32,
        )

        y = np.zeros(
            (
                self.batch_size,
                self.num_classes,
                *self.dim,
            ),
            dtype=np.float32,
        )

        for i, ID in enumerate(
            list_IDs_temp
        ):
            file_path = (
                self.base_dir / ID
            )

            if self.verbose == 1:
                print(
                    f"Training on: {file_path}"
                )

            if not file_path.is_file():
                raise FileNotFoundError(
                    f"HDF5 file not found: "
                    f"{file_path}"
                )

            with h5py.File(
                file_path, "r"
            ) as f:
                X[i] = np.asarray(
                    f["x"],
                    dtype=np.float32,
                )

                labels = np.asarray(
                    f["y"]
                )

                y[i] = np.moveaxis(
                    labels, 3, 0
                )[1:]

        return X, y

    def __getitem__(self, index):
        indexes = self.indexes[
            index * self.batch_size:
            (index + 1) * self.batch_size
        ]

        sample_list_temp = [
            self.sample_list[k]
            for k in indexes
        ]

        return self.__data_generation(
            sample_list_temp
        )


# =========================
# Image labeling
# =========================
def get_labeled_image(
    image,
    label,
    is_categorical=False,
):
    if not is_categorical:
        label = to_categorical(
            label,
            num_classes=4,
        ).astype(np.uint8)

    image = cv2.normalize(
        image[:, :, :, 0],
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    ).astype(np.uint8)

    labeled_image = np.zeros(
        label.shape[:3] + (3,),
        dtype=np.uint8,
    )

    background = label[
        :, :, :, 0
    ].astype(bool)

    for channel in range(3):
        labeled_image[
            :, :, :, channel
        ] = image * background

    tumor_mask = (
        label[:, :, :, 1:4] > 0
    )

    labeled_image[tumor_mask] = 255

    return labeled_image


# =========================
# Whole-volume prediction
# =========================
def predict_and_viz(
    image,
    label,
    model,
    threshold=0.5,
    loc=(100, 100, 50),
):
    if image.ndim != 4 or image.shape[-1] != 4:
        raise ValueError(
            "Expected MRI image with shape "
            "(X, Y, Z, 4). "
            f"Received: {image.shape}"
        )

    if label.shape != image.shape[:3]:
        raise ValueError(
            "Label dimensions must match "
            "the MRI spatial dimensions."
        )

    image_labeled = get_labeled_image(
        image.copy(),
        label.copy(),
    )

    sx, sy, sz = image.shape[:3]

    patch_x = 160
    patch_y = 160
    patch_z = 16

    model_label = np.zeros(
        (3, sx, sy, sz),
        dtype=np.float32,
    )

    for x in range(0, sx, patch_x):
        for y in range(0, sy, patch_y):
            for z in range(
                0, sz, patch_z
            ):
                patch = np.zeros(
                    (
                        4,
                        patch_x,
                        patch_y,
                        patch_z,
                    ),
                    dtype=np.float32,
                )

                volume = image[
                    x:x + patch_x,
                    y:y + patch_y,
                    z:z + patch_z,
                    :,
                ]

                p = np.moveaxis(
                    volume, -1, 0
                )

                dx, dy, dz = p.shape[1:]

                patch[
                    :, :dx, :dy, :dz
                ] = p

                pred = model.predict(
                    np.expand_dims(
                        patch, axis=0
                    ),
                    verbose=0,
                )

                model_label[
                    :,
                    x:x + dx,
                    y:y + dy,
                    z:z + dz,
                ] = pred[
                    0, :, :dx, :dy, :dz
                ]

    predicted_mask = (
        model_label >= threshold
    ).astype(np.uint8)

    predicted_mask = np.moveaxis(
        predicted_mask, 0, -1
    )

    model_label_reformatted = (
        np.zeros(
            (sx, sy, sz, 4),
            dtype=np.uint8,
        )
    )

    model_label_reformatted[
        :, :, :, 1:4
    ] = predicted_mask

    model_label_reformatted[
        :, :, :, 0
    ] = (
        np.sum(
            predicted_mask,
            axis=-1,
        ) == 0
    ).astype(np.uint8)

    model_labeled_image = (
        get_labeled_image(
            image,
            model_label_reformatted,
            is_categorical=True,
        )
    )

    x, y, z = loc

    if not (
        0 <= x < sx
        and 0 <= y < sy
        and 0 <= z < sz
    ):
        raise ValueError(
            "Visualization location "
            "is outside image dimensions."
        )

    fig, ax = plt.subplots(
        2, 3,
        figsize=(10, 7),
    )

    ax[0][0].imshow(
        np.rot90(
            image_labeled[x, :, :, :]
        )
    )
    ax[0][0].set_ylabel(
        "Ground Truth",
        fontsize=15,
    )
    ax[0][0].set_xlabel(
        "Sagittal",
        fontsize=15,
    )

    ax[0][1].imshow(
        np.rot90(
            image_labeled[:, y, :, :]
        )
    )
    ax[0][1].set_xlabel(
        "Coronal",
        fontsize=15,
    )

    ax[0][2].imshow(
        image_labeled[:, :, z, :]
    )
    ax[0][2].set_xlabel(
        "Transversal",
        fontsize=15,
    )

    ax[1][0].imshow(
        np.rot90(
            model_labeled_image[
                x, :, :, :
            ]
        )
    )
    ax[1][0].set_ylabel(
        "Prediction",
        fontsize=15,
    )

    ax[1][1].imshow(
        np.rot90(
            model_labeled_image[
                :, y, :, :
            ]
        )
    )

    ax[1][2].imshow(
        model_labeled_image[
            :, :, z, :
        ]
    )

    fig.subplots_adjust(
        wspace=0,
        hspace=0.12,
    )

    for i in range(2):
        for j in range(3):
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])

    return model_label_reformatted
