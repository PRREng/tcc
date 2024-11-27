import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm


def train_one_epoch(model, data_loader: DataLoader, criterion,
                    optimizer: Optimizer, epoch: int,
                    device: torch.device):
    model.train()
    total_loss = 0.0
    avg_acc = 0.0
    accumulation_steps = 4
    loop = tqdm(data_loader, desc=f"Epoch {epoch + 1}", unit="batch")
    for i, (x, y) in enumerate(loop):

        x, y = x.to(device), y.to(device)
        # forward pass
        y_hat = model(x)

        # calc loss
        loss = criterion(y_hat, y)
        total_loss += loss.item()

        # backward pass
        loss.backward()

        # zero grad and move gradient
        if (i+1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        preds = torch.argmax(y_hat, dim=-1)
        acc = (preds == y).sum().item() / len(y)
        avg_acc += acc

        loop.set_postfix(loss=total_loss / (i + 1),
                         accuracy=100. * avg_acc / (i + 1))
        del loss, y_hat

    total_loss /= len(data_loader)
    avg_acc /= len(data_loader)
    return total_loss, avg_acc


def test_one_epoch(model, data_loader: DataLoader, criterion,
                   epoch: int, device: torch.device):
    model.eval()
    total_loss = 0.0
    avg_acc = 0.0
    loop = tqdm(data_loader, desc=f"Epoch {epoch + 1}", unit="batch")
    with torch.no_grad():
        for i, (x, y) in enumerate(loop):

            x, y = x.to(device), y.to(device)
            # forward pass
            y_hat = model(x)

            # calc loss
            loss = criterion(y_hat, y)
            total_loss += loss.item()

            preds = torch.argmax(y_hat, dim=-1)
            acc = (preds == y).sum().item() / len(y)
            avg_acc += acc

            loop.set_postfix(loss=total_loss / (i + 1),
                             accuracy=100. * avg_acc / (i + 1))
            del loss, y_hat

    total_loss /= len(data_loader)
    avg_acc /= len(data_loader)
    return total_loss, avg_acc
