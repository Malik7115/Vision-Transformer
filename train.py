import torch
from torch.utils.data import DataLoader
from data import transformerDataset
from config import dataset_path, batch_size, lr, class_maps


dataset = transformerDataset('/home/ibrahim/Projects/Datasets/ViT')
print(dataset[1][0].shape)


train, test = torch.utils.data.random_split(dataset, [67,17])

print(len(train))
print(len(test))
# train_loader = DataLoader(dataset)