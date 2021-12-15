import os
import glob

from torch.utils.data import Dataset, DataLoader
import torchvision

from config import patch_size
from einops import rearrange
import cv2


os.system('clear')
class_maps = {"butterfly": 0, "ant": 1}

class transformerDataset(Dataset):
    def __init__(self, data_path, transform = None):
        
        self.image_paths = []
        self.labels      = []

        for i, path in enumerate(glob.glob(data_path + '/*', recursive=True)):

            files = os.listdir(path)
            for file in files:
                self.image_paths.append(path + '/' + file)
                self.labels.append(i)
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.resize(image, (100,100))
        label = self.labels[idx]

        image = rearrange(image, '(h p1) (w p2) c -> (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
        print(self.image_paths[idx])
        return (image, label)
        







