# Dataset Params
dataset_path = '/home/ibrahim/Projects/Datasets/ViT'
class_maps = {0 : "butterfly", 1 : "ant"}
patch_size = 7
num_patches = 16
totalClasses = 10
num_channels = 1


use_mnist = True

# Model Params
attention_heads = 2
embedding_dims  = 300

linear_dim      = 150
linear1_dim     = 150

# Training Params
shuffle = True
batch_size = 100
lr = 1e-4
total_epochs = 60


