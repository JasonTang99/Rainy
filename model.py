import tensorflow
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# base_model = MobileNet(weights = 'imagenet', include_top = False)
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
# x = Dense(1024,activation='relu')(x) #dense layer 2
# x = Dense(512,activation='relu')(x) #dense layer 3
# preds = Dense(120,activation='softmax')(x) #final layer with softmax activation
#
# model = Model(inputs=base_model.input,outputs=preds)
#
# for i,layer in enumerate(model.layers):
# 	print(i,layer.name)
# #
# for layer in base_model:
# 	layer.trainable = False

model = ResNet50(weights = 'imagenet')

img = image.load_img('./elephant.jpg', target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])