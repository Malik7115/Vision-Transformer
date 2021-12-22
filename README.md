# Vision-Transformer (ViT)


## Description
A simple model based on the vision transformer architecture from the paper, [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). 
The model is presented as a condensed form of the original model, to serve as a learning example.


## Architecture
There are three blocks in total:

* **Linear Projection**: Gives a linear projection of the pathes gathered from the input image

* **Multi-Headed Self Attention (MSA)**: The linear projections for each patch are concatenated with embedding vectors and an additional class embedding vector. These are then fed into a block of MSA. For simplicity there are only two heads for the MSA. 

* **Multi Layer Perceptron (MLP)**: The outputs corresponding to the class embedding vectors are sent to the MLP. The MLP consists of a series of fully connected layers. At the very last layer of the MLP the loss is calculated.

* **Skip Connections**: Skip connections were also incorporated to prevent exploding/vanishing gradients.


## Training
To keep things simple, most of the architecture is kept immutable. Refer to config.py to adjust parameters.
To test on mnist dataset, ```use_mnist = true```.
To test on custom dataset, ```use_mnist = false```.


*IMPORTANT:* Both the train and test samples must be divisible by the batch size. At the moment the inference/test for an individual image is not catered for, this will be done in a future commit.

To execute train loop simply run  
                ```python3 train.py```

## Results
With the default parameters in config.py, this model was able to achieve an train accuracy of 99.11% and test accuracy of 97.50% on MNIST dataset.

