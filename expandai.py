import yaml
import hreader
import dataset2
import torch

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from vit import ViT
import matplotlib.pyplot as plt
from epoch import train_one_epoch, test_one_epoch
import argparse

# from cnn import CNN

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="name of the model")
ap.add_argument("-l", "--load", default=False, help="load the model")
args = vars(ap.parse_args())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# carregue os dados formatados
X, y = hreader.getData()
# separe os dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=24
)

# crie o dataset
train_dataset = dataset2.MyDataSet(X_train, y_train)
test_dataset = dataset2.MyDataSet(X_test, y_test)

# crie o dataloader
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=1024, shuffle=True,
                          pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32,
                         shuffle=False, pin_memory=True)
print(f"Train_loader size: {len(train_loader)}")
print(f"Test_loader size: {len(test_loader)}")

# Construct the argument parser
model_name = args["model"]
with open(f"{model_name}.yml", "r") as file:
    config = yaml.safe_load(file)

hyperparams = config["hyperparams"]

model = ViT(**hyperparams).to(device)
# model = CNN()
# now load the model params
if args["load"]:
    model.load_state_dict(torch.load(f"{model_name}.pth", weights_only=True))
    print("loaded model")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

num_epochs = 50

train_losses = []
test_losses = []
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, epoch, device
    )

    val_loss, val_acc = test_one_epoch(model, test_loader, criterion,
                                       epoch, device)

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
