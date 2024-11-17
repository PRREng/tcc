from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, features, labels, transform=None,
                 target_transform=None):
        super(MyDataSet, self).__init__()
        self.features = features
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        sample = self.features[index, :, :]
        annotation = self.labels[index]  # there's 5 classes
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            annotation = self.target_transform(annotation)

        return sample, annotation
