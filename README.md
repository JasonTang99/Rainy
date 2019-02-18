# Rainy

Created using PyTorch to test and play around with Transfer Learning. 

Utilizes the pretrained Resnet152 model trained on image net from the pytorch model zoo. For more information on this model <a href="https://medium.com/@14prakash/understanding-and-implementing-architectures-of-resnet-and-resnext-for-state-of-the-art-image-cf51669e1624">here</a>.

Trained with ~500 images of rain images and ~900 images of non-rain images. In both cases 64% of images are used for training, 16% for validation and 20% for testing. The algorithm only sees training and validation data while learning and the test set is used to calculate the following statistics:

