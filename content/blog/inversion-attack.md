---
title: "Implementing a simple ML model inversion attack"
date: 2022-05-21T10:00:00+02:00
draft: false
---

With the increased use of machine learning applications, concerns about user privacy are emerging. In this post, I will demonstrate a model inversion (MI) attack that can reconstruct the input image solely from encoded information within a neural network.

Why are inversion attacks problematic? Consider the medical field as an example. Assume the physicians are [detecting tumors in CT scan images](https://wjso.biomedcentral.com/articles/10.1186/s12957-021-02259-6) using machine learning algorithms. Furthermore, if the physicians submit the [inference tasks to a central node](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjejoWWm_D3AhV0SfEDHWgADqAQFnoECAMQAQ&url=https%3A%2F%2Fwww.mdpi.com%2F2072-4292%2F12%2F17%2F2670%2Fpdf&usg=AOvVaw2ZqXKQnqMXxfNs-66w3pAx) to complete the computation, the information can be intercepted by an adversary who can reconstruct the input and therefore breach the patient's privacy.

To demonstrate how simple it is to reconstruct the input from encoded information, we will define two models: the target model and the attacking model. We will specify two parts for the target model. The adversary will capture the output of the first part and exploit it as input to the attacking model. This split architecture simulates distributed forward pass inference, which occurs across numerous network nodes (in our case it will be 2 nodes).

In reality, the attacker has access to the model and can send inputs to the target model to retrieve the corresponding encoding (first-part network output) and the final result (output of the second part of the network). This means that the attacker can collect training data by submitting a sequence of images to the target model and then capturing the encodings. These encodings will subsequently be used as inputs to the attacking model, and the images that were previously used as inputs to the target model will become the targets.

Let's move on to coding the attack in Python with `torch`. For demonstration purposes, our target model will perform the classification of handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

## Step 1: Importing libraries

We start by importing all the libraries we will need for executing the code.

```
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
```

## Step 2: Target model definition

Our target model will be divided into two parts. The first part will use three convolutional layers, while the second part will use two fully connected layers. To make the transition between the two parts as easy as possible, we need to create a function for flattening the encoding from the convolutional layers.

```
class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)
```

Now can define the target model as follows.

```
class ClassifierNN(nn.Module):
    def __init__(self):
        super(ClassifierNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=6, 
                kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=6, out_channels=12, 
                kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=12, out_channels=12, 
                kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
        )
        
        self.second_part = nn.Sequential(
            Flatten(),
            nn.Linear(588, 500),
            nn.LeakyReLU(),
            nn.Linear(500, 10),
            nn.Softmax(dim=-1),
        )
        
    def forward(self, x):
        f = self.first_part(x)
        return self.second_part(f)
        
        
model = ClassifierNN().cuda()
```

## Step 3: Training the target model

With the target model defined, we can train it. Since we will be using the MNIST dataset, we can use `torch` and download the data simply through its functions. For simplicity, we will use the testing data for training the attacking model later. 

```
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

dataset1 = datasets.MNIST(
    './data', train=True, download=True, transform=transform)
dataset2 = datasets.MNIST(
    './data', train=False, download=True, transform=transform)
    
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=32)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=16)
```

Now we can proceed to the training itself.

```
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
criterion = nn.NLLLoss()

model.train()

for epoch in range(3):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    scheduler.step()
```

## Step 4: Attack model definition

To reconstruct the input image from the target model encoding we will use three 2D transposed convolution operators.

```
class AttackerNN(nn.Module):
    def __init__(self):
        super(AttackerNN, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=12, out_channels=12, 
                kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                in_channels=12, out_channels=6, 
                kernel_size=3, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                in_channels=6, out_channels=1, 
                kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        
    def forward(self, x):
        return self.layers(x)
        
        
attacker_model = AttackerNN().cuda()
```

## Step 5: Training the attacking model

With the model defined, we can train it. Notice that we are using the outputs of the first part of the target model as the inputs.

```
optimiser = optim.Adam(attacker_model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(10):
    for batch_idx, (data, _) in enumerate(test_loader):
        data = data.cuda()

        optimiser.zero_grad()

        target_outputs = model.first_part(data)
        attack_outputs = attacker_model(target_outputs)

        loss = criterion(data, attack_outputs)

        loss.backward()
        optimiser.step()
```

## Step 6: Evaluation

The successfully trained attacking model can now reconstruct the inputs to the target model purely on its encoding. You can see an example of this attack below.

![Attack evaluation](/attack.png)

## Conclusion
While the model inversion attack described and demonstrated above does not pose a substantial danger because classification is not tested across several computing nodes in simple machine learning applications, it does demonstrate a vulnerability that machine learning models face. More complex models and applications where the privacy of sensitive information is essential [must be protected](https://arxiv.org/abs/2201.10787) from such attacks.