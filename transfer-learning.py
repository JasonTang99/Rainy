from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import copy

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, save_data=False, load_state_path="", checkpoint_path=""):
    # Trys to load in previous model state if path is given
    try:
        if load_state_path is not "":
            previous_state = torch.load(load_state_path)
            model.load_state_dict(previous_state['model_state_dict'])
            optimizer.load_state_dict(previous_state['optimizer_state_dict'])
            epoch_count = previous_state['epoch_count']

            # print(previous_state['model_state_dict'])
            print(previous_state['optimizer_state_dict'])
            print("Previous State loaded without error")
        else:
            epoch_count = 0
    except Exception as e:
        print("Model could not be loaded from " + load_state_path)
        print(e)
        epoch_count = 0

    if not frozen_params:
        for param in model.parameters():
            param.requires_grad = True


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + epoch_count + 1, num_epochs + epoch_count))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / num_samples_train
            epoch_acc = running_corrects.double() / num_samples_train


            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if save_data is True:
                fw = open("stats.txt", "a+")
                fw.write('{} {} {:.4f} {:.4f}\n'.format(epoch + epoch_count + 1, phase, epoch_loss, epoch_acc))

            # Saves a checkpoint every 5 epochs and the last epoch, but not the first epoch
            if phase == 'val' and (epoch == num_epochs-1 or epoch%5 == 0 and epoch != 0):
                try: 
                    torch.save({'epoch_count': epoch_count+epoch+1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, checkpoint_path)
                    print("Checkpoint saved")
                except:
                    print("Checkpoint failed to save")
        print()

    return model

def visualize_model(model, num_images=6):
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    return

def set_data_transform():
    return {
        'train': transforms.Compose([
            transforms.RandomRotation(10, expand=True),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.08, 0.08, 0.08, 0.08),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

def display_model_params():
    params = list(model.parameters())
    print(len(params))
    for param in params:
        print(param.size())
        if param.requires_grad is True:
            print("TRAINABLE")

def run_test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloaders['test']:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy of the network on the test images: " + str(100.0 * correct / total) + "%")

# Sets up configuration
training = True
testing = True
num_samples_train = 750
num_samples_test = 1000
save_model = True
load_model = False
frozen_params = True

# PATH_CHECKPOINT = ""
PATH_CHECKPOINT = "./models/checkpoint18.tar"
PATH_SAVE = "./models/res18.pth"



# Sets up proper data transformations
data_transforms = set_data_transform()

# Sets the directory to get the test, val, and train data
data_dir = 'rain_data'

# Sets up getters for test, val, and train data
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ['train', 'val', 'test']
}
dataloaders = {
    x: torch.utils.data.DataLoader(
        dataset=image_datasets[x], 
        batch_size=2, 
        shuffle=False, 
        num_workers=4,
        sampler=torch.utils.data.RandomSampler(
            data_source=image_datasets[x], 
            replacement=True,
            num_samples=num_samples_train
        )
    )
    for x in ['train', 'val']
}
# Set this one to have a batch size of 1 so test runs through all images
dataloaders['test'] = torch.utils.data.DataLoader(
    dataset=image_datasets['test'], 
    batch_size=1, 
    shuffle=False, 
    num_workers=4,
    # sampler=torch.utils.data.RandomSampler(
    #     data_source=image_datasets['test'], 
    #     replacement=True,
    #     num_samples=num_samples_test
    # )
)

# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

# class_names = image_datasets['train'].classes

# Sets up which device (CPU or GPU) to use
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on CUDA GPU")
else:
    device = torch.device("cpu")
    print("Running on CPU")
print()

if load_model:
    model = torch.load(PATH_SAVE)

if training:
    # Loads in the pretrained model and freezes its parameters
    model = models.resnet18(pretrained=True)
    if frozen_params:
        for param in model.parameters():
            param.requires_grad = False
    
    # Resets the top layers
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    # display_model_params()

    # Sets up training parameters
    model = model.to(device)
    criterion = nn.CrossEntropyLoss() 
    optimizer_ft = optim.Adam(model.parameters(), lr=0.001)
    decay = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model = train_model(
        model, 
        criterion, 
        optimizer_ft, 
        decay, 
        num_epochs = 20, 
        save_data = True,
        checkpoint_path=PATH_CHECKPOINT,
        load_state_path=PATH_CHECKPOINT
    )

    if save_model:
        torch.save(model, PATH_SAVE)

if testing:

    # Run on test set to get proper stats
    run_test()

# plt.ion()

# visualize_model(model, 20)
# plt.ioff()
# plt.show()
