from keras.applications import inception_v3
from keras import backend as K

model = inception_v3.InceptionV3(weights='imagenet',
                                 include_top=False)

model.summary()

from keras import models
from keras import layers
model = models.load_model("models/simple_mnist.h5")
model.pop()
model.summary()
