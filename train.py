import torch
from torch.utils.data import DataLoader
from data import transformerDataset
from tqdm import tqdm
from config import dataset_path, batch_size, lr, class_maps, total_epochs, use_mnist, patch_size, test_interval
from model import ViT

from torchvision import datasets
from torchvision.transforms import ToTensor
from einops import rearrange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if use_mnist:
    train = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
    )
    test = datasets.MNIST(
        root = 'data', 
        train = False, 
        transform = ToTensor()
    )
    
    train_dataloader = DataLoader(train, batch_size = batch_size, shuffle = True, num_workers = 4)
    test_dataloader  = DataLoader(test,  batch_size = batch_size, shuffle = True, num_workers = 4)

else:
    dataset = transformerDataset(dataset_path)
    print(dataset[1][0].shape)
    train, test = torch.utils.data.random_split(dataset, [256,20])
    train_dataloader = DataLoader(train, batch_size = batch_size, shuffle = True, num_workers = 4)
    test_dataloader  = DataLoader(test,  batch_size = batch_size, shuffle = True, num_workers = 4)

model = ViT().to(device)
totalParams = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


for epoch in tqdm(range(total_epochs)):
    acc = 0
    train_acc = 0
    running_loss = 0

    for patch, label in train_dataloader:
        if (use_mnist):
            patch    = rearrange(patch, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)

        patch = patch.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        output = model(patch)
        loss   = criterion(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * patch.size(0)

        resutls = output
        resutls = torch.nn.functional.softmax(resutls)
        resutls = torch.max(resutls, dim = -1)[1]
        train_acc += torch.sum((resutls == label.to(device))/batch_size)

    running_loss = running_loss/len(train_dataloader)
    train_acc    = train_acc/len(train_dataloader)

    if(epoch % test_interval == 0):
        for test_batch in test_dataloader:
            test_out = rearrange(test_batch[0], 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
            resutls = model(test_out.to(device))
            resutls = torch.nn.functional.softmax(resutls)
            resutls = torch.max(resutls, dim = -1)[1]
            acc += torch.sum((resutls == test_batch[1].to(device))/batch_size)
        running_acc  = acc/len(test_dataloader)
        print("loss: ",running_loss, "train acc", train_acc, "test acc: ", running_acc)
    
    else:
        print("loss: ",running_loss, "train acc", train_acc)
