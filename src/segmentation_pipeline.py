
import numpy as np
import pandas as pd
import nibabel as nib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from pathlib import Path

import util


# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "BraTS-Data"
MODEL_PATH = DATA_DIR / "model_pretrained.hdf5"


# =========================
# Data loading
# =========================
def load_case(image_nifty_file, label_nifty_file):
    image = np.asarray(
        nib.load(str(image_nifty_file)).get_fdata()
    )
    label = np.asarray(
        nib.load(str(label_nifty_file)).get_fdata()
    )
    return image, label


# =========================
# Patch extraction
# =========================
def get_sub_volume(
    image,
    label,
    orig_x=240,
    orig_y=240,
    orig_z=155,
    output_x=160,
    output_y=160,
    output_z=16,
    num_classes=4,
    max_tries=1000,
    background_threshold=0.95,
):
    tries = 0

    while tries < max_tries:
        start_x = np.random.randint(0, orig_x - output_x + 1)
        start_y = np.random.randint(0, orig_y - output_y + 1)
        start_z = np.random.randint(0, orig_z - output_z + 1)

        y = label[
            start_x:start_x + output_x,
            start_y:start_y + output_y,
            start_z:start_z + output_z,
        ]

        y = tf.keras.utils.to_categorical(
            y, num_classes=num_classes
        )

        bgrd_ratio = (
            np.sum(y[:, :, :, 0])
            / (output_x * output_y * output_z)
        )

        tries += 1

        if bgrd_ratio < background_threshold:
            X = np.copy(
                image[
                    start_x:start_x + output_x,
                    start_y:start_y + output_y,
                    start_z:start_z + output_z,
                    :,
                ]
            )

            X = np.moveaxis(X, -1, 0)
            y = np.moveaxis(y, -1, 0)

            y = y[1:, :, :, :]
            return X, y

    print(
        f"Tried {tries} times to find a sub-volume. "
        "Giving up..."
    )
    return None, None


# =========================
# Preprocessing
# =========================
def standardize(image):
    standardized_image = np.zeros_like(
        image, dtype=np.float32
    )

    for c in range(image.shape[0]):
        for z in range(image.shape[3]):
            image_slice = image[c, :, :, z]
            centered = image_slice - np.mean(image_slice)

            std = np.std(centered)

            if std != 0:
                centered_scaled = centered / std
            else:
                centered_scaled = centered

            standardized_image[c, :, :, z] = centered_scaled

    return standardized_image


# =========================
# Metrics and loss
# =========================
def single_class_dice_coefficient(
    y_true,
    y_pred,
    axis=(0, 1, 2),
    epsilon=1e-5,
):
    dice_numerator = (
        2 * K.sum(y_true * y_pred, axis=axis) + epsilon
    )
    dice_denominator = (
        K.sum(y_true, axis=axis)
        + K.sum(y_pred, axis=axis)
        + epsilon
    )
    return dice_numerator / dice_denominator


def dice_coefficient(
    y_true,
    y_pred,
    axis=(1, 2, 3),
    epsilon=1e-5,
):
    dice_numerator = (
        2 * K.sum(y_true * y_pred, axis=axis) + epsilon
    )
    dice_denominator = (
        K.sum(y_true, axis=axis)
        + K.sum(y_pred, axis=axis)
        + epsilon
    )
    return K.mean(dice_numerator / dice_denominator)


def soft_dice_loss(
    y_true,
    y_pred,
    axis=(1, 2, 3),
    epsilon=1e-5,
):
    dice_numerator = (
        2 * K.sum(y_true * y_pred, axis=axis) + epsilon
    )
    dice_denominator = (
        K.sum(K.square(y_pred), axis=axis)
        + K.sum(K.square(y_true), axis=axis)
        + epsilon
    )
    return 1 - K.mean(
        dice_numerator / dice_denominator
    )


# =========================
# Evaluation
# =========================
def compute_class_sens_spec(pred, label, class_num):
    class_pred = pred[class_num]
    class_label = label[class_num]

    tp = np.sum(
        (class_pred == 1) & (class_label == 1)
    )
    tn = np.sum(
        (class_pred == 0) & (class_label == 0)
    )
    fp = np.sum(
        (class_pred == 1) & (class_label == 0)
    )
    fn = np.sum(
        (class_pred == 0) & (class_label == 1)
    )

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    return sensitivity, specificity


def get_sens_spec_df(pred, label):
    metrics_df = pd.DataFrame(
        columns=[
            "Edema",
            "Non-Enhancing Tumor",
            "Enhancing Tumor",
        ],
        index=["Sensitivity", "Specificity"],
    )

    for i, class_name in enumerate(metrics_df.columns):
        sens, spec = compute_class_sens_spec(
            pred, label, i
        )
        metrics_df.loc[
            "Sensitivity", class_name
        ] = round(sens, 4)
        metrics_df.loc[
            "Specificity", class_name
        ] = round(spec, 4)

    return metrics_df


# =========================
# Model
# =========================
def build_model():
    model = util.unet_model_3d(
        loss_function=soft_dice_loss,
        metrics=[dice_coefficient],
    )
    return model


# =========================
# Inference helpers
# =========================
def predict_patch(model, X):
    X_norm = standardize(X)
    X_batch = np.expand_dims(X_norm, axis=0)
    return model.predict(X_batch)


def prepare_whole_scan_labels(label, pred):
    whole_scan_label = keras.utils.to_categorical(
        label, num_classes=4
    )

    whole_scan_label = np.moveaxis(
        whole_scan_label, 3, 0
    )[1:4]

    whole_scan_pred = np.moveaxis(
        pred, 3, 0
    )[1:4]

    return whole_scan_pred, whole_scan_label


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    image_path = (
        DATA_DIR / "imagesTr" / "BRATS_003.nii.gz"
    )
    label_path = (
        DATA_DIR / "labelsTr" / "BRATS_003.nii.gz"
    )

    for path in [image_path, label_path, MODEL_PATH]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    image, label = load_case(
        image_path, label_path
    )

    if image.ndim != 4 or image.shape[-1] != 4:
        raise ValueError(
            "Expected image shape (240, 240, 155, 4), "
            f"but got {image.shape}. "
            "Check that the MRI modalities are stacked."
        )

    X, y = get_sub_volume(image, label)

    if X is None:
        raise RuntimeError(
            "Could not extract a valid sub-volume."
        )

    X_norm = standardize(X)

    model = build_model()
    model.load_weights(str(MODEL_PATH))

    patch_pred = model.predict(
        np.expand_dims(X_norm, axis=0)
    )

    patch_pred_binary = (
        patch_pred[0] > 0.5
    ).astype(np.uint8)

    patch_metrics = get_sens_spec_df(
        patch_pred_binary, y
    )
    print(patch_metrics)
