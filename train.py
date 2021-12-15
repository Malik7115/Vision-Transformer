import torch
from torch.utils.data import DataLoader
from data import transformerDataset
from tqdm import tqdm
from config import dataset_path, batch_size, lr, class_maps, total_epochs
from model import ViT

model = ViT()
totalParams = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

dataset = transformerDataset('/home/ibrahim/Projects/Datasets/ViT')
print(dataset[1][0].shape)


train, test = torch.utils.data.random_split(dataset, [64,20])
train_dataloader = DataLoader(train, batch_size = batch_size, shuffle = True, num_workers = 4)
test_dataloader  = DataLoader(test,  batch_size = batch_size, shuffle = True, num_workers = 4)

test_batch = next(iter(train_dataloader))
print(test_batch[1])


for epoch in range(total_epochs):
    running_loss = 0
    for patch, label in tqdm(train_dataloader):
        patch.cuda()
        label.cuda()
        
        optimizer.zero_grad()
        output = model(patch)
        loss   = criterion(output, label)
        loss.backward()
        optimizer.step()

        test_out = model(test_batch[0])
        
        running_loss += loss.item() * patch.size(0)
        
    running_loss = running_loss / len(train_dataloader)
    print(running_loss)
    
    
print(">>>>>>>>>>>>>>>>>>>> RESULTS >>>>>>>>>>>>>>")
print(test_batch[1])
resutls = torch.nn.functional.softmax(test_out)
print(torch.max(resutls, dim = -1)[1])



print(len(train))
print(len(test))
# train_loader = DataLoader(dataset)