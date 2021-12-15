import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from config import totalClasses, attention_heads, batch_size

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.embed = embed_dim
        self.Q     = nn.Linear(embed_dim, embed_dim, bias = False)
        self.K     = nn.Linear(embed_dim, embed_dim, bias = False)
        self.V     = nn.Linear(embed_dim, embed_dim, bias = False)
    
    def forward(self, x):
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        #transpose K for dot product with Q
        attn_scores = torch.bmm(Q, K.transpose(1,2)) 
        #Softmax across embedding (column) dim
        attn_scores  = attn_scores.softmax(dim = 2)

        attn_weights = torch.bmm(attn_scores, V)

        return attn_weights

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(1200, 512)
        self.fc2  = nn.Linear(512, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
        


class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.classes     = totalClasses
        # self.classEmbed  = nn.Embedding(self.classes, 150)
        # self.classEmbed  = 
        self.classEmbed  = repeat(nn.Parameter(torch.randn(1,150)), '() e -> b e', b=batch_size)
        self.posEmbed    = nn.Embedding(100+1, 150)
        self.LinearProj  = torch.nn.Linear(300, 150)
        self.LinearProj1 = torch.nn.Linear(900, 300)

        self.batchNorm    = nn.BatchNorm1d(101)
        self.batchNorm1   = nn.BatchNorm1d(101)

        self.MSA = []
        for i in range(attention_heads):
            self.MSA.append(SelfAttention(300))

        self.MLP = MLP()


    def forward(self, x): # x are flattened patches and y is label

        x = x.type(torch.float32)
        x = self.LinearProj(x) # 100,150
        x = nn.functional.relu(x)

        posEmbeddings  = self.posEmbed(torch.tensor(range(101)).repeat(16,1)) # 101,150

        
        classEmbedding = torch.cat((self.classEmbed, posEmbeddings[:,0,:]), dim = -1) # 0,300
        x = torch.cat((x, posEmbeddings[:,1:,:]), dim = -1) # 100,300

        x = torch.cat((classEmbedding.unsqueeze(dim = 1), x), dim = 1) # 101,300

        skip1 = x.clone().detach()
        MSA_outputs = []

        # x = x.unsqueeze(dim = 0) # remove after checking

        for SA in self.MSA:
            MSA_outputs.append(SA(x))

        MSA_outputs = torch.cat(MSA_outputs, dim = -1)

        x = self.batchNorm(MSA_outputs)
        x = torch.cat((x, skip1), dim = -1)

        skip2 = x.clone().detach()

        x = self.LinearProj1(x)
        x = nn.functional.relu(x)

        x = self.batchNorm1(x)
        x = torch.cat((x,skip2), dim = -1)
        

        x = self.MLP(x[:,0,:])

        return x
