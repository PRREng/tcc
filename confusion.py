import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, \
        roc_curve, auc
from model import ViT

# Load test data from the .h5 file
test_data_path = "test_data.h5"
with h5py.File(test_data_path, "r") as f:
    # Access datasets
    x_np = np.array(f["x_data"][:])
    y_np = np.array(f["y_data"][:])


x_tensor = torch.tensor(x_np,
                        dtype=torch.float32
                        ).permute(0, 2, 1).contiguous()
y_tensor = torch.tensor(y_np, dtype=torch.long)

# Data augmentation
real = x_tensor[:, :, 0]
imaginary = x_tensor[:, :, 1]
magnitude = torch.sqrt(real**2 + imaginary**2)
phase = torch.atan2(imaginary, real)

# append magnitude and phase to features
features = torch.cat((x_tensor, magnitude.unsqueeze(2),
                     phase.unsqueeze(2)), dim=2)

x_tensor = features
print(x_tensor.shape)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the saved PyTorch model
model_name = "vit_r13_200"
hyperparams = {
    "in_features": 4,
    "d_model": 25,
    "seq_len": 100,
    "num_heads": 5,
    "ff_d": 1024,
    "num_layers": 7,
    "num_classes": 5,
    "dropout": 0.1
}
model = ViT(**hyperparams).to(device)
# model = CNN()
# now load the model params
model.load_state_dict(torch.load(f"{model_name}.pth",
                                 map_location=device,
                                 weights_only=True))

model.eval()  # Set the model to evaluation mode

# Make predictions
with torch.no_grad():
    logits = model(x_tensor)  # Forward pass through the model
    y_pred_probs = torch.softmax(logits, dim=1).numpy()
    y_pred = np.argmax(y_pred_probs, axis=1)  # Predicted class indices

# Convert true labels to one-hot encoding
num_classes = y_pred_probs.shape[1]
y_true_onehot = np.eye(num_classes)[y_tensor.numpy()]

# Plot confusion matrix
cm = confusion_matrix(y_tensor.numpy(), y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[f'C{i+1}'
                                              for i in range(num_classes)])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()

# Plot ROC curves
plt.figure(figsize=(10, 8))
for i in range(num_classes):  # Iterate over each class
    fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'C{i+1} (AUC = {roc_auc:.2f})')

# Plot diagonal and labels
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc='lower right')
plt.show()
