# Accuracy of the network on the test images: 94.9820788530466%

import matplotlib.pyplot as plt
import numpy as np

f = open("stats.txt", "r")
line = f.readline()

data_train_acc = []
data_train_loss = []
data_val_acc = []
data_val_loss = []

while line is not '':
	line = line[0:-1]
	line_split = line.split(" ")

	if line_split[1] == 'train':
		data_train_acc.append(float(line_split[3]))
		data_train_loss.append(float(line_split[2]))
	else:
		data_val_acc.append(float(line_split[3]))
		data_val_loss.append(float(line_split[2]))

	line = f.readline()

# print(data_train_acc)
# print(len(data_train_acc))
# print(data_train_loss)
# print(data_val_acc)
# print(data_val_loss)


fig, ax = plt.subplots()
ax.plot(data_val_acc, label='Validation Accuracy')
ax.plot(data_train_acc, label='Training Accuracy')
ax.axis([-5, 205, 0.60, 1.0])

ax.set(xlabel='Epoch', ylabel='Accuracy',
       title='Training Using Resnet18')
ax.grid()
ax.legend()

fig.savefig("test.png")
plt.show()