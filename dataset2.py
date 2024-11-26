from torch.utils.data import Dataset
import torch
import numpy as np


class MyDataSet(Dataset):
    def __init__(self, features, labels, transform=None,
                 target_transform=None):
        super(MyDataSet, self).__init__()
        # calculate magnitude and phase
        real = features[:, :, 0]
        imaginary = features[:, :, 1]
        magnitude = np.sqrt(real**2 + imaginary**2)
        phase = np.arctan2(imaginary, real)

        # append magnitude and phase to features
        features = np.concatenate((features, magnitude[:, :, None],
                                   phase[:, :, None]), axis=2)

        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        sample = self.features[index, :, :]
        annotation = self.labels[index]  # there's 5 classes
        sample = sample.view(1, 100, 4)
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            annotation = self.target_transform(annotation)
        sample = sample.squeeze()

        return sample, annotation
