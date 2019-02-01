from keras.applications import MobileNet
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import keras

base_model = MobileNet(weights = 'imagenet', include_top = False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x = Dense(1024,activation='relu')(x) #dense layer 2
x = Dense(512,activation='relu')(x) #dense layer 3
preds = Dense(120,activation='softmax')(x) #final layer with softmax activation

model = Model(inputs=base_model.input,outputs=preds)

for i,layer in enumerate(model.layers):
	print(i,layer.name)