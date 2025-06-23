# https://stackoverflow.com/questions/70078251/cannot-run-carlini-and-wagner-attack-using-foolbox-on-a-tensorflow-model

import tensorflow as tf
import numpy as np

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cityblock
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from foolbox import TensorFlowModel
from foolbox.criteria import Misclassification
from foolbox.attacks import L2CarliniWagnerAttack

num_classes = 12

print("Step 1: Load model and weights")
baseModel = ResNet50(weights=None, include_top=False, input_tensor=Input(shape=(224, 224, 3)))
headModel = baseModel.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(num_classes, activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)

bounds = (0, 1)
fmodel = TensorFlowModel(model, bounds=bounds)
images, labels = tf.random.normal((64, 10, 224, 224, 3)), tf.random.uniform((64, 10,), maxval=13, dtype=tf.int32)


for i in range(0, images.shape[0]):
    print("document "+str(i))
    features_to_test=images[i,:]
    features_to_test=(features_to_test/6+1)/2
    print(tf.reduce_min(features_to_test), tf.reduce_max(features_to_test))
    labels_to_test=labels[i,:]
    epsilons = np.linspace(0.01, 1, num=2)
    attack = L2CarliniWagnerAttack()
    adversarials = attack(fmodel, features_to_test, criterion=Misclassification(labels_to_test), epsilons=epsilons)