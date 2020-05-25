from torchvision import datasets, transforms, models
import torch
import numpy as np
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F
from PIL import Image

from collections import OrderedDict
import os

# Constants
CROP_SIZE = 224
RESIZE = 256
MEANS = [0.485, 0.456, 0.406]
ST_DEV = [0.229, 0.224, 0.225]
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DROPOUT = 0.3
HIDDEN_LAYER = 512
EPOCHS = 6
use_gpu = torch.cuda.is_available()
pre_models = {"vgg16": 25088,
              "densenet121": 1024,
              "alexnet": 9216}


def load_data(data_dir='flowers'):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'training': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(CROP_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEANS, ST_DEV)
        ]),
        'validation': transforms.Compose([
            transforms.Resize(RESIZE),
            transforms.CenterCrop(CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MEANS, ST_DEV)
        ]),
        'testing': transforms.Compose([
            transforms.Resize(RESIZE),
            transforms.CenterCrop(CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MEANS, ST_DEV)
        ]),
    }

    image_datasets = {
        'training': datasets.ImageFolder(train_dir, transform=data_transforms['training']),
        'testing': datasets.ImageFolder(test_dir, transform=data_transforms['testing']),
        'validation': datasets.ImageFolder(valid_dir, transform=data_transforms['validation'])
    }

    dataloaders = {
        'training': torch.utils.data.DataLoader(image_datasets['training'], batch_size=BATCH_SIZE, shuffle=True),
        'testing': torch.utils.data.DataLoader(image_datasets['testing'], batch_size=BATCH_SIZE, shuffle=True),
        'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size=BATCH_SIZE, shuffle=True)
    }
    return dataloaders, image_datasets


def get_model(structure='densenet121', dropout=DROPOUT, hidden_layer1=HIDDEN_LAYER):
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif structure == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        print("The model is not expected".format(structure))

    for param in model.parameters():
        param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(dropout)),
            ('inputs', nn.Linear(pre_models[structure], hidden_layer1)),
            ('relu1', nn.ReLU()),
            ('hidden_layer1', nn.Linear(hidden_layer1, 256)),
            ('relu2', nn.ReLU()),
            ('hidden_layer2', nn.Linear(256, 128)),
            ('relu3', nn.ReLU()),
            ('hidden_layer3', nn.Linear(128, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

        model.classifier = classifier
        model.cuda()

        return model


def training(model, criterion, optimizer, training_loader, validation_loader, epochs=EPOCHS):
    # Putting the model into training mode
    model.train()
    print_every = 10
    steps = 0

    # Iterates through each training pass based on #epochs & GPU/CPU
    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in iter(training_loader):
            steps += 1
            inputs = Variable(inputs.float().cuda())
            labels = Variable(labels.long().cuda())
            # Forward and backward passes
            optimizer.zero_grad()
            output = model.forward(inputs)  # Forward propogation
            loss = criterion(output, labels)  # Calculates loss
            loss.backward()  # Calcule the gradient
            optimizer.step()  # Update the weights based on the gradient and the learning rate
            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss, accuracy = validating(model, criterion, validation_loader)
                print("Epoch: {}/{} ".format(epoch + 1, epochs),
                      "Training Loss: {:.3f} ".format(running_loss / print_every),
                      "Validation Loss: {:.3f} ".format(validation_loss),
                      "Validation Accuracy: {:.3f}".format(accuracy))


def validating(model, criterion, data_loader):
    test_loss = 0
    accuracy = 0
    # Set validation mode to the model
    model.eval()

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = Variable(inputs.float().cuda())
            labels = Variable(labels.long().cuda())
            output = model.forward(inputs)
            test_loss += criterion(output, labels).item()
            ps = torch.exp(output).data
            equality = (labels.data == ps.max(1)[1])
            accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss / len(data_loader), accuracy / len(data_loader)


def train_process(args):
    dataloaders, image_datasets = load_data(args.data_dir)

    model = get_model(args.arch, DROPOUT, args.hidden_units)

    if args.gpu:
        if use_gpu:
            model = model.cuda()
            print("Using GPU: " + str(use_gpu))
        else:
            print("Using CPU since GPU is not available/configured")

    criterion = nn.CrossEntropyLoss()  # nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), args.learning_rate)

    training(model, criterion, optimizer, dataloaders['training'], dataloaders['validation'], args.epochs)
    # validation_criterion = nn.NLLLoss()
    # validating(model, validation_criterion, dataloaders['validation'])

    model.class_to_idx = image_datasets['training'].class_to_idx
    checkpoint = {'epochs': args.epochs,
                  'batch_size': BATCH_SIZE,
                  'classifier': model.classifier,
                  'structure': args.arch,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),  # Holds all the weights and biases
                  'class_to_idx': model.class_to_idx,
                  'dropout': DROPOUT,
                  'hidden_units': args.hidden_units
                  }
    if args.save_dir:
        checkpoint_path = os.path.join(args.save_dir, args.saved_model)
    else:
        checkpoint_path = args.saved_model

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(path):
    checkpoint = torch.load(path)

    structure = checkpoint['structure']
    model = get_model(structure, checkpoint['dropout'], checkpoint['hidden_units'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    model.classifier = checkpoint['classifier']

    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']

    for param in model.parameters():
        param.requires_grad = False

    return model, optimizer, epochs


def process_image(image):
    img_pil = Image.open(image)

    adjustments = transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEANS, std=ST_DEV)
    ])

    img_tensor = adjustments(img_pil)

    return img_tensor


def predict(image_path, model, topk=5, use_gpu=True):
    if use_gpu:
        model = model.cuda()
    model.eval()
    img_tensor = process_image(image_path)
    img_tensor = img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.float()

    with torch.no_grad():
        output = model.forward(img_tensor.cuda())

    probability = F.softmax(output.data, dim=1)
    return probability.topk(topk)


def extract_classes(cat_to_name, probabilities_tensor):
    classes = [cat_to_name[str(index + 1)] for index in np.array(probabilities_tensor[1][0])]
    probabilities = np.array(probabilities_tensor[0][0]).tolist()
    return classes, probabilities
