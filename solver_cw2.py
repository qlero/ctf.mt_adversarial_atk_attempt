import os
import numpy as np
# import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import tifffile
from art.attacks.evasion import CarliniL0Method as cw
from PIL import Image
from art.estimators.classification import KerasClassifier

from foolbox import TensorFlowModel
from foolbox.criteria import TargetedMisclassification
from foolbox.attacks import L2CarliniWagnerAttack

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print("\n"*10)

from pathlib import Path
from collections import OrderedDict
from typing import Callable, Tuple, Optional, Union

import torch
import numpy as np
import pandas as pd
import tensorflow as tf


DIM_NN = {
    'batch': 0,
    'depth': 1,
    'height': 2,
    'width': 3,
    'channels': 4,
}

DIM_PT = {
    'batch': 0,
    'channels': 1,
    'depth': 2,
    'height': 3,
    'width': 4,
}


def transpose_to_pytorch(array: np.ndarray) -> np.ndarray:
    """
    See docs of torch.nn.Conv3d
    https://pytorch.org/docs/stable/nn.html#conv3d
    """
    shape = (
        DIM_NN['batch'],
        DIM_NN['channels'],
        DIM_NN['depth'],
        DIM_NN['height'],
        DIM_NN['width'],
    )
    array = np.transpose(array, shape)
    return array


def transpose_to_tensorflow(array: np.ndarray) -> np.ndarray:
    """
    See docs of tf.nn.conv3d
    https://www.tensorflow.org/api_docs/python/tf/nn/conv3d#args
    """
    shape = (
        DIM_PT['batch'],
        DIM_PT['depth'],
        DIM_PT['height'],
        DIM_PT['width'],
        DIM_PT['channels'],
    )
    array = np.transpose(array, shape)
    return array


def niftynet_batch_to_torch_tensor(batch_dict: dict) -> torch.Tensor:
    window = batch_dict['image']
    window = window[..., 0, :]  # remove time dimension
    window = transpose_to_pytorch(window)
    # https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
    tensor = torch.from_numpy(window.copy())
    return tensor


def torch_logits_to_niftynet_labels(logits: torch.Tensor) -> np.ndarray:
    logits = logits.detach().cpu()
    labels = logits.argmax(dim=DIM_PT['channels'], keepdim=True).numpy()
    labels = labels.astype(np.uint16)
    labels = transpose_to_tensorflow(labels)
    return labels


def tf2pt(
        name_tf: str,
        array_tf: np.ndarray,
        name_mapping_function: Callable,
        ) -> Tuple[str, np.ndarray]:
    name_pt = name_mapping_function(name_tf)
    num_dimensions = array_tf.dim()
    if num_dimensions == 1:
        array_pt = array_tf
    elif num_dimensions == 5:
        array_pt = array_tf.permute(4, 3, 0, 1, 2)
    else:
        raise NotImplementedError
    return name_pt, array_pt


"""
This last couple of functions are a good reminder to myself that
TensorFlow makes me sad and PyTorch makes me happy
"""


def checkpoint_tf_to_state_dict_tf_(
        input_checkpoint_tf_path: Union[str, Path],
        output_csv_tf_path: Union[str, Path],
        output_state_dict_tf_path: Union[str, Path],
        filter_out_function: Optional[Callable] = None,
        replace_string: Optional[str] = None,
        ) -> None:
    tf.reset_default_graph()

    rows = []
    variables_dict = OrderedDict()
    variables_list = tf.train.list_variables(str(input_checkpoint_tf_path))
    for name, shape in variables_list:
        if filter_out_function is not None and filter_out_function(name, shape):
            continue
        variables_dict[name] = tf.get_variable(name, shape=shape)
        if replace_string is not None:
            name = name.replace(replace_string, '')
        shape = ', '.join(str(n) for n in shape)
        row = {'name': name, 'shape': shape}
        rows.append(row)
    data_frame = pd.DataFrame.from_dict(rows)

    saver = tf.train.Saver()
    state_dict = {}
    with tf.Session() as sess:
        saver.restore(sess, str(input_checkpoint_tf_path))
        for name, shape in variables_list:
            if (
                    filter_out_function is not None
                    and filter_out_function(name, shape)
                    ):
                continue
            array = variables_dict[name].eval()
            if replace_string is not None:
                name = name.replace(replace_string, '')
            state_dict[name] = torch.tensor(array)

    data_frame.to_csv(output_csv_tf_path)
    print(data_frame)
    torch.save(state_dict, output_state_dict_tf_path)


def checkpoint_tf_to_state_dict_tf(*args, **kwargs) -> None:
    """
    This is done so that the GPU can be used by PyTorch afterwards
    https://stackoverflow.com/a/44842044/3956024

    If you don't need to run the inference, call
    checkpoint_tf_to_state_dict_tf_ instead to avoid potential headaches

    This might break things on Windows if trying to run as a Python file.
    See https://github.com/pytorch/pytorch/issues/5858#issuecomment-373950687
    """
    import multiprocessing
    p = multiprocessing.Process(
        target=checkpoint_tf_to_state_dict_tf_,
        args=args,
        kwargs=kwargs,
    )
    p.start()
    p.join()


import re
from typing import Tuple


def is_not_valid(variable_name: str, shape: Tuple[int, ...]) -> bool:
    exclusion_criteria = (
        'Adam' in variable_name,  # used for training
        'biased' in variable_name,  # unused
        not shape,  # empty variables
        'ExponentialMovingAverage' in variable_name,  # unused on NiftyNet model zoo
    )
    return any(exclusion_criteria)


def tf2pt_name(name_tf: str) -> str:
    """
    Return the equivalent PyTorch parameter name of the TensorFlow
    variable. Rules have been created from visual inspection of the
    variables lists.
    """
    param_type_dict = {
        'w': 'weight',
        'gamma': 'weight',
        'beta': 'bias',
        'moving_mean': 'running_mean',
        'moving_variance': 'running_var',
    }

    if name_tf.startswith('res_'):
        # For example: 'res_2_0/bn_0/moving_variance'
        pattern = (
            'res'
            r'_(\d)'   # dil_idx = 2
            r'_(\d)'   # res_idx = 0
            r'/(\w+)'  # layer_type = bn
            r'_(\d)'   # layer_idx = 0
            r'/(\w+)'  # param_type = moving_variance
        )
        groups = re.match(pattern, name_tf).groups()
        dil_idx, res_idx, layer_type, layer_idx, param_type = groups
        param_idx = 3 if layer_type == 'conv' else 0

        name_pt = (
            f'block.{dil_idx}.dilation_block.{res_idx}.residual_block'
            f'.{layer_idx}.convolutional_block.{param_idx}.{param_type_dict[param_type]}'
        )
    elif name_tf.startswith('conv_'):
        conv_layers_dict = {
            'conv_0_bn_relu/conv_/w': 'block.0.convolutional_block.1.weight',  # first conv layer
            'conv_0_bn_relu/bn_/gamma': 'block.0.convolutional_block.2.weight',
            'conv_0_bn_relu/bn_/beta': 'block.0.convolutional_block.2.bias',
            'conv_0_bn_relu/bn_/moving_mean': 'block.0.convolutional_block.2.running_mean',
            'conv_0_bn_relu/bn_/moving_variance': 'block.0.convolutional_block.2.running_var',

            'conv_1_bn_relu/conv_/w': 'block.4.convolutional_block.0.weight',  # layer with dropout
            'conv_1_bn_relu/bn_/gamma': 'block.4.convolutional_block.1.weight',
            'conv_1_bn_relu/bn_/beta': 'block.4.convolutional_block.1.bias',
            'conv_1_bn_relu/bn_/moving_mean': 'block.4.convolutional_block.1.running_mean',
            'conv_1_bn_relu/bn_/moving_variance': 'block.4.convolutional_block.1.running_var',

            'conv_2_bn/conv_/w': 'block.6.convolutional_block.0.weight',  # classifier
            'conv_2_bn/bn_/gamma': 'block.6.convolutional_block.1.weight',
            'conv_2_bn/bn_/beta': 'block.6.convolutional_block.1.bias',
            'conv_2_bn/bn_/moving_mean': 'block.6.convolutional_block.1.running_mean',
            'conv_2_bn/bn_/moving_variance': 'block.6.convolutional_block.1.running_var',
        }
        name_pt = conv_layers_dict[name_tf]
    else:
        raise NotImplementedError
    return name_pt

############################
# declares important paths #
############################

model_weights = "./model/mt_classifier_ft_0.h5"
classes_txt = "./model/classes.txt"
source_image = "./hotdog.png"
save_image = "./upload.tiff"

###############################
# Declares important variables #
################################

flag_label = 101

check=1

###############################
# Defines important functions #
###############################

def compare_to_hotdog(img_path):
        path = "./hotdog.png"
        img = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = preprocess_input(x)
        x_input = np.expand_dims(x, axis=0)

        path = img_path
        img = tifffile.imread(path)

        if img.dtype == np.float32:
            img = img
        elif img.dtype == np.uint16:
            img = img.astype(np.float32) / 256.0
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if len(img.shape) == 2:
            # print("grayscale to rgb")
            img = np.stack([img, img, img], axis=2)
            # print("after conversion:", img.shape)
        elif len(img.shape) == 3:
            # print(f"found {img.shape[2]} channels")
            if img.shape[2] == 1:
                # print("grayscale to rgb")
                img = np.concatenate([img, img, img], axis=2)
                # print("after conversion:", img.shape)
            elif img.shape[2] == 2:
                # print("grayscale + alpha to rgb")
                grayscale = img[:, :, 0:1]
                img = np.concatenate([grayscale, grayscale, grayscale], axis=2)
                # print("after conversion:", img.shape)
            elif img.shape[2] != 3:
                # print(f"unexpected number of channels: {img.shape[2]}")
                img = img[:, :, :3]
            elif img.shape[2] == 4:
                # print("found 4 channels, taking 3")
                img = img[:, :, 0:3]
                # print("after reduction:", img.shape)
            else:
                # print(f"unexpected {img.shape[2]} channels")
                if img.shape[2] >= 3:
                    img = img[:, :, :3]
                else:
                    first_channel = img[:, :, 0:1]
                    img = np.concatenate([first_channel, first_channel, first_channel], axis=2)
                # print("after adjustment:", img.shape)
        from scipy.ndimage import zoom
        if img.shape[:2] != (224, 224):
            zoom_factors = (224 / img.shape[0], 224 / img.shape[1], 1.0)
            img = zoom(img, zoom_factors, order=1)
            # prin1t("after resize:", img.shape)
        if img.max() > 2.0:
            x = preprocess_input(img)
        else: 
            x = (img / 127.5) - 1.0
        a_input = np.expand_dims(x, axis=0)

        diff_pixels = np.abs(a_input - x_input)
        l2_distance = np.linalg.norm((a_input - x_input).flatten())

        return l2_distance, diff_pixels

def preprocess(img):
    """
    Processess image from uint8 to [-1, 1] (normalized) image of size 224x224
    """
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img


def unprocess(img, height, width):
    """
    Processess image from [-1, 1] (normalized) image of size 224x224 to uint8 image of size heightxwidth
    """
    # img = tf.image.resize(img, (height, width))
    img = np.array(img)[0]
    img = (img + 1)*(255/2)
    img = img.astype(np.uint8)
    return img

def read_to_np(path):
    """
    Imports an image from disk to numpy array
    """
    img = image.load_img(path, target_size=(224, 224))
    img = image.img_to_array(img)
    height, width, _ = img.shape
    return img, height, width

def write_to_disk(img, name):
    """
    Write numpy array to disk.
    """
    # tf.keras.utils.save_img(name, img)
    print(np.array(img).shape)
    img = Image.fromarray(np.array(img), 'RGB')
    img.save(f'{name}')


def clip(tensor, c):
    """
    Clips tensorflow array to [-c, c]
    """
    # clip the values of the tensor to a given range and return it
    return tf.clip_by_value(tensor, clip_value_min=-c, clip_value_max=c)
            

def normalize_image(img):
    grads_norm = img[:,:,0]+ img[:,:,1]+ img[:,:,2]
    grads_norm = (grads_norm - tf.reduce_min(grads_norm))/ (tf.reduce_max(grads_norm)- tf.reduce_min(grads_norm))
    return grads_norm


def carlini_wagner(model, benign_img, delta, start_class, target_class, loss, steps, epsilon, ratio, mask = None):
    """
    Performs a *targeted* PGD attack [1].

    [1] Madry et al. Towards Deep Learning Models Resistant to Adversarial Attacks
        https://arxiv.org/pdf/1706.06083
    """
    # mod = KerasClassifier(model=model, clip_values=(-1,1))
    # attack = cw(mod)
    # adv = attack.generate(benign_img)
    import torchattacks
    atk1 = torchattacks.CW(model, c=0.3, kappa=10, steps=1000, lr=0.001)
    atk1.targeted = True
    adv_images = atk1(benign_img, target_class)
    return True

################################
# Generates adversarial attack #
################################

# Loads class names
with open(classes_txt) as f:
    classes = [i.strip() for i in f.readlines()]
print([(i, classes[i]) for i in range(len(classes))])


# Loads image
img, height, width = read_to_np(source_image)
print(f"Image of width {width} and height {height}.")

# Loads model
model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    weights=None,
    classes=102,
    classifier_activation='softmax'
)

model           = load_model(model_weights)
# model.load_weights(model_weights)
# model.trainable = False # freeze weights
print(model.summary())


checkpoint_tf_to_state_dict_tf(
    input_checkpoint_tf_path=model_weights,
    output_csv_tf_path="variables_tf.csv",
    output_state_dict_tf_path="state_dict_tf.pth",
    filter_out_function=is_not_valid,
    replace_string='MobileNetV2/',
)



data_frame_tf = pd.read_csv("variables_tf.csv")
state_dict_tf = torch.load("state_dict_tf.pth")
print(data_frame_tf)


# # Infers on benign image
# img_array  = preprocess(keras.utils.img_to_array(img))
# img_preds  = model.predict(img_array)
# benign_pred_index = np.argmax(img_preds)
# benign_pred_label = classes[benign_pred_index]
# print(f" Predicted label on benign image is {benign_pred_label} @ index {benign_pred_index}")
# if benign_pred_index != 55:
#    raise ValueError("Benign image is not properly setup for the CTF.")



# # Computes the adversarial perturbation
# val = True
# while True:

#     print("#"*30)
#     print("RESTART")
#     print("#"*30)

#     epsilon = 2/255
#     steps = 200
#     lr = 0.005
#     ratio = 0.01

#     print(steps, epsilon, lr, ratio)

#     # Generates the learning process to produce the adversarial example
#     optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
#     loss = tf.keras.losses.SparseCategoricalCrossentropy()

#     label = tf.one_hot(flag_label, img_preds.shape[-1])
#     label = tf.reshape(label, (1, img_preds.shape[-1]))

#     benign_img = tf.constant(img_array, dtype=tf.float32)
#     delta      = tf.Variable(tf.zeros_like(benign_img), trainable=True)
    
#     val = carlini_wagner(model, benign_img, delta, benign_pred_index, flag_label, loss, steps, epsilon, ratio)
#     if not val:
#         break