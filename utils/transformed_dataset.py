import torch as tc


class TransformedDataset(tc.utils.data.Dataset):
    '''将 transform 施加在 dataset 上'''

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.dataset[index])

    def __len__(self):
        return len(self.dataset)
