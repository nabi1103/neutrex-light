import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('../neutrex')
import gdl
from gdl.datasets.ImageTestDataset import CustomTestData, TestData
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Link to the training dataset.
img_dir = "/home/stthnguye/dataset/affectnet/images"
# Link to the annotation files
annotations_file = "/home/stthnguye/neutrex-lite/assets/finetune/tensors/csv/ee_labels_affectnet_50000.csv"
# Link to the precomputed output vector from the original encoder
precomp_dir = "/home/stthnguye/neutrex-lite/assets/finetune/tensors/affectnet-reduced/"
# Link to the experiment directory, which contain the encoder to be finetuned.
experiment_dir = "/home/stthnguye/neutrex-lite/experiment/finetune-ee-0.7-large/"
# Name for the validation plot
plot_name = "Finetuning pruned model EE-0.7 with AffectNet"
# Config for training
BATCH_SIZE = 64
EPOCH = 10
train_size = 30000

dataset = CustomTestData(img_dir, face_detector="fan", max_detection=1, label_path = annotations_file)
dataset_small = torch.utils.data.Subset(dataset, range(33000))
test_size = len(dataset_small) - train_size

train_set, val_set = torch.utils.data.random_split(dataset_small, [train_size, test_size])

# Check the training for bad images 
def check_tensor_dim(data):
    for img in data:
        if (img["image"].shape != torch.Size([1,3,224,224])):
            print(img["image_name"])
            img["image"] = torch.unsqueeze(img["image"], 0).to(device)
    images = torch.stack([img["image"] for img in data]).to(device)
    image_names = [img["image_name"] for img in data]
    image_paths = [img["image_path"] for img in data]
    label_paths = [img["label_path"] for img in data]
    result = {
        "image" : images,
        "image_name": image_names,
        "image_path": image_paths,
        "label_path": label_paths
    }
    return result

def finetune_ee(model):
    data_loader_train = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=check_tensor_dim
    )

    data_loader_val = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        collate_fn=check_tensor_dim
    )

    train(EPOCH, model, data_loader_train, data_loader_val, plot_name)

def train_one(model, train_data):
    loss_fn = torch.nn.MSELoss()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scaler = GradScaler()

    for e in enumerate(train_data):
        batch_idx = e[0]
        images, targets = e[1]["image"], e[1]["label_path"]
        
        optimizer.zero_grad()
        l = len(images)

        current_size = BATCH_SIZE

        total_loss = 0
        for i in range(l):
            try:
                output = model(images[i])
                # target = torch.load(targets[i])
                target = torch.load(precomp_dir + targets[i] + "_ee.pt")
                total_loss = loss_fn(output, target) + total_loss
                del output
            except:
                current_size = current_size - 1
                continue

        total_loss = total_loss / current_size
        total_loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print("Current batch: " + str(batch_idx) + ", current loss: " + str(float(total_loss)))

def train(epoch, model, train_data, val_data, plot_name):
    val_scores = []
    
    for e in range(epoch):
        train_one(model, train_data)
        print("Epoch " + str(e) + " finished")
        val_score = validate(model, val_data)
        print("Validation score: ", val_score)
        val_scores.append(val_score)
    
    # Where to save the model
    torch.save(model, experiment_dir + "pruned-finetuned-ee.pth")
    plot_training(val_scores, "Finetuning pruned model EE-0.7")


def validate(model, val_data):
    loss_fn = torch.nn.MSELoss()
    total_loss = 0
    model.eval()
    current_size = test_size

    for e in enumerate(val_data):
        images, targets = e[1]["image"], e[1]["label_path"][0]
        l = len(images)
        for i in range(l):
            try:
                output = model(images[i])
                target = torch.load(precomp_dir + targets[i] + "_ee.pt")
                total_loss = float(loss_fn(output, target)) + total_loss
            except:
                current_size = current_size - 1
                continue

    return total_loss / current_size

def plot_training(val_scores, name):
    plt.plot(val_scores, 'b-o', color = "black", label = "Validation score")
    plt.xticks(np.arange(len(val_scores)), np.arange(1, len(val_scores)+1))
    plt.xlabel("Epoch")
    plt.ylabel("Validation score")
    plt.legend(loc="upper right")
    plt.title(label=name, fontsize=16, color="black")
    plt.savefig(os.path.join(experiment_dir, "validation_score"))

# Load the pruned encoder and finetune
model = torch.load(experiment_dir + "original/pruned-ee.pth")
finetune_ee(model)
