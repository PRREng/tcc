import yaml
import hreader
import dataset
import torch

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from vit import ViT
from epoch import test_one_epoch
import argparse

# from cnn import CNN

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="name of the model")
args = vars(ap.parse_args())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# carregue os dados formatados
X, y = hreader.getData()
# separe os dados
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.25, random_state=24
)

# crie o dataset
test_dataset = dataset.MyDataSet(X_test, y_test)

# crie o dataloader
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
print(f"Test_loader size: {len(test_loader)}")

# Construct the argument parser
model_name = args["model"]
with open(f"{model_name}.yml", "r") as file:
    config = yaml.safe_load(file)

hyperparams = config["hyperparams"]

model = ViT(**hyperparams).to(device)
# model = CNN()
# now load the model params
model.load_state_dict(torch.load(f"{model_name}.pth", weights_only=True))
print("loaded model")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

val_loss, val_acc = test_one_epoch(model, test_loader, criterion,
                                   1, device)

print(f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")
