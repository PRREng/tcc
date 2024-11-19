# import util
import yaml
import hreader
import dataset
import torch
# import torchvision.transforms as tt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from vit import ViT
import matplotlib.pyplot as plt
from epoch import train_one_epoch, test_one_epoch
import argparse
# from tqdm import tqdm
# from cnn import CNN

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="name of the model")
args = vars(ap.parse_args())

# carregue os dados formatados
X, y = hreader.getData()
# separe os dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=24
)

# crie o dataset
train_dataset = dataset.MyDataSet(X_train, y_train)
test_dataset = dataset.MyDataSet(X_test, y_test)

# crie o dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
print(f"Train_loader size: {len(train_loader)}")
print(f"Test_loader size: {len(test_loader)}")

# Construct the argument parser
model_name = args["model"]
with open("vit_r8.yml", "r") as file:
    config = yaml.safe_load(file)

hyperparams = config["hyperparams"]

model = ViT(**hyperparams)
# model = CNN()
# now load the model params
model.load_state_dict(torch.load(f"{model_name}.pth", weights_only=True))
print("loaded model")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

num_epochs = 310

train_losses = []
test_losses = []
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                            optimizer, epoch)

    val_loss, val_acc = test_one_epoch(model, test_loader,
                                       criterion, epoch)

    train_losses.append(train_loss)
    test_losses.append(val_loss)

print(f"Train loss: {train_losses[-1]}, acc: {train_acc}")
print(f"Val loss: {test_losses[-1]}, acc: {val_acc}")

# plot the loss and val loss
plt.plot(range(num_epochs), train_losses, color="red", label="train_loss")
plt.plot(range(num_epochs), test_losses, label="val_loss")
plt.show()

torch.save(model.state_dict(), f"{model_name}.pth")
print("Model saved successfully.")
