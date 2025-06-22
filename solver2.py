import os
import numpy as np
# import matplotlib.pyplot as plt
from keras.models import load_model
import keras
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import tifffile


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print("\n"*10)

############################
# declares important paths #
############################

model_weights = "./model/mt_classifier_ft_0.h5"
classes_txt = "./model/classes.txt"
source_image = "./hotdog.png"
save_image = "./upload.tiff"

################################
# Declares important variables #
################################

flag_label = 101

check=5

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
    img = img.numpy()[0]
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
    tf.keras.utils.save_img(name, img)


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

import matplotlib.pyplot as plt
cmap = 'Pastel1'
# plot the image 
def plot_maps(img1, img2,vmin=0.3,vmax=0.7, mix_val=2):
    f = plt.figure(figsize=(15,45))
    plt.subplot(1,3,1)
    plt.imsave("a.png", img1,vmin=vmin, vmax=vmax, cmap=cmap)
    plt.axis("off")
    plt.subplot(1,3,2)
    plt.imsave("b.png", img2, cmap = cmap)
    plt.axis("off")
    plt.subplot(1,3,3)
    plt.imsave("c.png", img1*mix_val+img2/mix_val, cmap = cmap)
    plt.axis("off")


def PGD_attack(model, benign_img, delta, start_class, target_class, loss, steps, epsilon, ratio, mask = None):
    """
    Performs a *targeted* PGD attack [1].

    [1] Madry et al. Towards Deep Learning Models Resistant to Adversarial Attacks
        https://arxiv.org/pdf/1706.06083
    """
    # visual help to find interesting areas
	# # Iteratively refines a perturbation array <delta>
    # with tf.GradientTape() as tape:
    #     tape.watch(benign_img)
    #     result = model(benign_img)
    #     max_idx = tf.argmax(result, axis = 1)
    #     max_score = result[0, max_idx[0]]
    # grads = tape.gradient(max_score, benign_img)
    # plot_maps(normalize_image(grads[0]), normalize_image(benign_img[0]))
    # print(normalize_image(grads[0]).numpy().min(), normalize_image(grads[0]).numpy().max(), grads.numpy().shape)

    from art.estimators.classification import KerasClassifier
    from art.attacks.evasion import FastGradientMethod, CarliniLInfMethod
    # tf.compat.v1.disable_eager_execution()
    # tf.compat.v1.disable_eager_execution()
    model.compile()
    model.run_eagerly = False
    classifier = KerasClassifier(model=model, clip_values=(-1, 1))
    attack_cw = CarliniLInfMethod(classifier=model,
                              max_iter=100,
                              learning_rate=0.01,
                              initial_const=1e0,
                              largest_const=2e0,
                              targeted=True)
    adv = attack_cw.generate(benign_img, target_class)


    predictions   = model(adv, training=False)

    write_to_disk(adv[0], f"adv_{check}.tiff")
    write_to_disk(adv[0], f"adv_{check}.png")
    write_to_disk(adv[0]-benign_img[0], f"diff_{check}.tiff")
    l2_distance, diff_pixels = compare_to_hotdog(f"adv_{check}.tiff")
    print(f"conf: {predictions[0, 101]:.4}, l2: {l2_distance.mean():.4}, Mean diff: {diff_pixels.mean():.4}, Max diff: {diff_pixels.max():.4}")

    return False

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
# write_to_disk(img, f"hotdog_check@load_{check}.tiff")


# Loads model
model           = load_model(model_weights)
model.trainable = False # freeze weights
print(model)

# Infers on benign image
img_array  = preprocess(keras.utils.img_to_array(img))
# img_preds  = model(img_array)
# print(img_preds)
# benign_pred_index = np.argmax(np.array(img_preds))
# benign_pred_label = classes[benign_pred_index]
# print(f" Predicted label on benign image is {benign_pred_label} @ index {benign_pred_index}")
# if benign_pred_index != 55:
#    raise ValueError("Benign image is not properly setup for the CTF.")



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
    
    val = PGD_attack(model, benign_img, delta, benign_pred_index, label, loss, steps, epsilon, ratio)
    if not val:
        break