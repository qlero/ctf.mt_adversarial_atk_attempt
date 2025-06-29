# !pip install numpy==1.26.4 eagerpy foolbox

import tensorflow as tf
import keras
import eagerpy as ep
from foolbox import TensorFlowModel, accuracy
from foolbox.attacks import L2CarliniWagnerAttack, LinfPGD, L2DeepFoolAttack, LinfinityBrendelBethgeAttack, L2BrendelBethgeAttack, SparseL1DescentAttack
from foolbox.criteria import TargetedMisclassification
from PIL import Image
import numpy as np
import os
import time
import argparse

with open("model/classes.txt", "r") as f:
  x = f.readlines()
  x = [i.strip() for i in x]

for f in x:
  if f == "flag":
      f = "z_flag"
  if not os.path.exists("dataset/"+f):
    os.makedirs("dataset/"+f)

# GLOBALS

start = time.time()

BOUNDS = (-1, 1)
PREPROCESSING = dict()
ATTACK_NAME = "L2CarliniWagnerAttack" # L2DeepFoolAttack #LinfPGD #L2CarliniWagnerAttack #L2BrendelBethgeAttack #LinfinityBrendelBethgeAttack, #SparseL1DescentAttack
# EPSILONS = np.linspace(0.02, 0.05, num=20) #for #LinfPGD
# ERROR WITH LinfinityBrendelBethgeAttack
# ERROR WITH L2BrendelBethgeAttack
# ERROR WITH L2BrendelBethgeAttack
# No convergence SparseL1DescentAttack
# No convergence L2DeepFoolAttack
EPSILONS = np.linspace(0.01, 2, num=10)
# EPSILONS = [0.0, 0.0002, 0.0005, 0.0008, 0.001, 0.0015, 0.002, 0.003, 0.01, 0.1, 0.3, 0.5, 1.0,]
SAVE_PATH = "adversarial_examples"

# FUNCTIONS

def load_model(bounds, preprocessing):
    model = keras.models.load_model("./model/mt_classifier_ft_0.h5")
    model.summary()
    model = TensorFlowModel(model, bounds=bounds, preprocessing=preprocessing)
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='solver adversarial attack')
    parser.add_argument('-c', '--cpu', action='store_true')
    args = parser.parse_args()

    try:
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        gpus = tf.config.list_physical_devices('GPU')
        gpus[0].name
        device = "/device:GPU:0"
    except:
        device = "/device:CPU:0"

    if args.cpu:
        device = "/device:CPU:0"

    try:
        os.remove("dataset/.ipynb_checkpoints")
    except:
        print("no file to delete")
    len(os.listdir("dataset"))

    start = time.time()

    print(f"[INFO] Running on device: {device}")
    with tf.device(device):

        #loads model
        model = load_model(BOUNDS, PREPROCESSING)

        print("\n"*3)

        #loads data
        data = tf.keras.utils.image_dataset_from_directory(
            "dataset", 
            validation_split=0, 
            seed=123, 
            image_size=(224, 224), 
            batch_size=1
        )
        for image_batch, label_batch in data:
            image, label = ep.astensors(image_batch, label_batch)
            image = (image-127.5) / 255 * 2

        print("\n"*3)

        #tests inference
        pred = model(image)
        shape = pred.shape
        prediction = pred.argmax(1)
        print(f"[INFO] OG image -- output size {shape}, output label: {prediction}")
        if prediction != 55:
            raise ValueError("[ERROR] Wrong inference on normal image")
        
        print("\n"*3)

        #runs an attack
        criterion = TargetedMisclassification(np.array([data.class_names.index("z_flag")]))
        print(criterion)
        print("[INFO] Running the attack")
        if ATTACK_NAME == "LinfPGD":
            attack = LinfPGD(rel_stepsize=0.001, steps=200)
        elif ATTACK_NAME == "L2CarliniWagnerAttack":
            attack = L2CarliniWagnerAttack(steps=10000, stepsize=1e-3, confidence=0)
        elif ATTACK_NAME == "LinfinityBrendelBethgeAttack":
            attack = LinfinityBrendelBethgeAttack()
        elif ATTACK_NAME == "L2BrendelBethgeAttack":
            attack = L2BrendelBethgeAttack()
        elif ATTACK_NAME == "SparseL1DescentAttack":
            attack = SparseL1DescentAttack()
        elif ATTACK_NAME == "L2DeepFoolAttack":
            attack = L2DeepFoolAttack()
        else:
            raise ValueError("[ERROR] wrong <ATTACK_NAME> provided")
        
        raw_advs, clipped_advs, success = attack(model, image, criterion, epsilons=EPSILONS)

        print("\n"*3)

        #checks results
        print("[INFO] Checking results")
        robust_accuracy = 1 - success.float32().mean(axis=-1)
        print("[INFO] robust accuracy for perturbations with")
        for eps, acc in zip(EPSILONS, robust_accuracy):
            print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")

        # we can also manually check this
        # we will use the clipped advs instead of the raw advs, otherwise
        # we would need to check if the perturbation sizes are actually
        # within the specified epsilon bound
        print("[INFO] robust accuracy for perturbations with")
        for eps, advs_ in zip(EPSILONS, clipped_advs):
            acc2 = accuracy(model, advs_, label)
            print(f"  Linf norm ≤ {eps:<6}: {acc2 * 100:4.1f} %")
            print("    perturbation sizes:")
            perturbation_sizes = (advs_ - image).norms.linf(axis=(1, 2, 3)).numpy()
            print("    ", str(perturbation_sizes).replace("\n", "\n" + "    "))
            if acc2 == 0:
                break

        print("\n"*3)      

        for i, img in enumerate(clipped_advs):
            pred = model(img)
            shape = pred.shape
            prediction = pred.argmax(1)
            print(f"[INFO] Adv. attack (+save) @ eps {EPSILONS[i]:.4f} -- input shape: " + \
                  f"{img.shape}, output size {shape}, output label: {prediction}")
            if prediction == 101:
                img = ((img + 1)/2 * 255).numpy().clip(0, 255).astype(np.uint8)
                img = Image.fromarray(img[0])
                img.save(f"{SAVE_PATH}/{ATTACK_NAME}_eps{EPSILONS[i]:4f}_target{prediction}.png")
                img.save(f"{SAVE_PATH}/{ATTACK_NAME}_eps{EPSILONS[i]:4f}_target{prediction}.tiff")

        for i, img in enumerate(clipped_advs):
            pred = model(img)
            shape = pred.shape
            prediction = pred.argmax(1)
            print(f"[INFO] Raw adv. attack (+save) @ eps {EPSILONS[i]:.4f} -- input shape: " + \
                  f"{img.shape}, output size {shape}, output label: {prediction}")
            if prediction == 101:
                img = ((img + 1)/2 * 255).numpy().clip(0, 255).astype(np.uint8)
                img = Image.fromarray(img[0])
                img.save(f"{SAVE_PATH}/{ATTACK_NAME}_raw_eps{EPSILONS[i]:4f}_target{prediction}.png")
                img.save(f"{SAVE_PATH}/{ATTACK_NAME}_raw_eps{EPSILONS[i]:4f}_target{prediction}.tiff")

    end = time.time()

    print("\n"*3)

    print(f"[INFO] Took {end - start}seconds")