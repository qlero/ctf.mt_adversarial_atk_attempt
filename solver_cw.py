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

    fmodel = TensorFlowModel(model, bounds=(-1,1), preprocessing=dict())
    epsilons = [0.5] #np.linspace(0.03, 1, num=1)
    print(epsilons)
    clone = np.array(benign_img)
    attack = L2CarliniWagnerAttack(steps=4, stepsize=1e-2, confidence=0.3, binary_search_steps=9)
    print(attack)
    crit = TargetedMisclassification(np.array([101]))
    print(crit)
    _, clipped_advs, success = attack(fmodel, benign_img, criterion=crit, epsilons=epsilons)
    print(success)
    for i, adv in enumerate(clipped_advs):
        write_to_disk(unprocess(adv, 224, 224), f"hotdog_check@cw_{i}_check_{check}.tiff")
        write_to_disk(unprocess(adv, 224, 224), f"hotdog_check@cw_{i}_check_{check}.png")
        write_to_disk(unprocess(adv-clone, 224, 224), f"hotdog_check@cw_{i}_diff_{check}.png")
        l2_distance, diff_pixels = compare_to_hotdog(f"hotdog_check@cw_{i}_check_{check}.tiff")
        print(l2_distance, diff_pixels.mean(), diff_pixels.max())
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

# Infers on benign image
img_array  = preprocess(keras.utils.img_to_array(img))
img_preds  = model.predict(img_array)
benign_pred_index = np.argmax(img_preds)
benign_pred_label = classes[benign_pred_index]
print(f" Predicted label on benign image is {benign_pred_label} @ index {benign_pred_index}")
if benign_pred_index != 55:
   raise ValueError("Benign image is not properly setup for the CTF.")



# Computes the adversarial perturbation
val = True
while True:

    print("#"*30)
    print("RESTART")
    print("#"*30)

    epsilon = 2/255
    steps = 200
    lr = 0.005
    ratio = 1

    print(steps, epsilon, lr, ratio)

    # Generates the learning process to produce the adversarial example
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    label = tf.one_hot(flag_label, img_preds.shape[-1])
    label = tf.reshape(label, (1, img_preds.shape[-1]))

    benign_img = tf.constant(img_array, dtype=tf.float32)
    delta      = tf.Variable(tf.zeros_like(benign_img), trainable=True)
    
    val = carlini_wagner(model, benign_img, delta, benign_pred_index, flag_label, loss, steps, epsilon, ratio)
    if not val:
        break