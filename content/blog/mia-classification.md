---
title: "Membership inference attacks on classification models"
date: 2022-07-08T15:31:00+02:00
draft: false
---

Machine learning has enabled a high degree of automation and data modelling in domains such as [biology](https://www.nature.com/articles/s41467-021-22518-0), [linguistics](https://arxiv.org/abs/2005.14165) and [medicine](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-019-0681-4). While the [first neural network](https://blogs.umass.edu/brain-wars/files/2016/03/rosenblatt-1957.pdf) was created by Frank Rosenblatt in 1957, the real applications of the networks became prominent only in recent years. Two main factors contribute to this advancement: increased computational resources and availability of data. Data can be accessed as easily as ever and is also being produced at an accelerating rate, which allows for creating bigger and better machine learning models. However, oftentimes the data contain private information, which can be the case for [medical applications](https://www.nature.com/articles/s42256-020-0186-1) or [recommendation systems](https://arxiv.org/abs/2102.04925). Even a brief experience with designing and testing machine learning models reveals that they are prone to overfitting, i.e., aligning the model too closely to the training data and therefore decreasing the ability to generalise (unless prevented by the expert). Overfitted models imprint the information of the training set into their weights, which leads to an [increased vulnerability](https://ieeexplore.ieee.org/abstract/document/8429311) to attacks and subsequent private information leaks. 

> The aim of a membership inference attack (MIA) on a model is to infer whether a data point was a member of its training dataset or not. 

The idea behind attacks on classification-based models is that they are *overconfident* with data samples that were included in the training dataset. During training, the model is optimised over multiple epochs and the training members are seen several times. Compared to testing data that the model sees for the first time the returned probability distribution is usually less poised and more balanced. This imposes a privacy vulnerability on the training dataset members. 

For example, the adversary can target a model trained on [medical records](https://proceedings.mlr.press/v143/gupta21a.html) to predict a probability of a certain condition. Once inferring the target model with specific medical information the adversary can reveal whether that particular patient was associated with the medical condition or not violating the patient's privacy. 

MIAs are usually separated into three main types.

* **Black-box attack:** The most restrictive attack from the view of the adversary. The only data that can be accessed from the target model is the final prediction. In classification tasks, it's usually the class probability distribution coming from the SoftMax activation layer. The most restrictive setting assumes the target model returns [only the predicted class label](https://arxiv.org/abs/2007.14321). Moreover, the adversary shouldn't have any information regarding the model architecture or the dataset used to train the model.
* **White-box attack:** The opposite of the black-box attack where the adversary has the full knowledge about the used architecture, the training dataset, and mainly the outputs of each model's layers.
* **Grey-box attack:** Anything that doesn't fall under the two categories above can be considered a grey-box attack.

Currently, there are two main approaches used to attack the target model. 

* **Threshold attack:** The most intuitive way of exposing an input `x` as a training sample is to measure the maximum prediction confidence. If the confidence exceeds a [predefined threshold](https://arxiv.org/abs/1806.01246), implying the model overconfidence, the input is considered a member of the training dataset. 
* **Binary classifier:** The attack is carried on by a binary classifier trained on outputs of a *shadow model* that imitates the functionality of the *target model*. The shadow model is trained by the adversary to behave as close to the target model as possible. This, however, is dependent on the type of attack. In the black-box setting, the knowledge about the target model architecture or the training hyperparameters is kept from the adversary making the attack more difficult. In grey-box or even white-box settings, the adversary can more closely replicate the target model with the shadow model, leading to better performance. Once the shadow model is trained, the attacker collects its outputs by inferring it on training and testing data and assembles a training dataset for the attack model. The trained attack model can then be used to infer the outputs of the target model.

While it is possible to do a MI attack with a single shadow model, a higher number of shadow models correlates with a better attack performance.

The [main factor](https://arxiv.org/abs/2009.05669) that contributes to the success of the attack is model overfitting. A consequence of overfitting is the gap in performance on training and testing data, which can arise from the high complexity of the target model or the limited size of the training dataset. This is something the adversary can exploit. The defence mechanism can therefore be regularisation or early stopping which prevents overfitting. It might result in a lower utility of the model, but with increased privacy and resistance to membership inference attacks.

The type of the target model influences the attack outcome as well. While the attacks are carried mainly on deep models, other machine learning models are susceptible to the MI attack as well. The highest attack accuracy is obtained on models that are especially sensitive to outliers. By contrast, models that don't change the decision boundary easily are less prone to privacy breaches. For this reason, decision trees are the most vulnerable as a unique feature in the training data can result in a completely new branch. Naive Bayes models are, on the other hand, [affected the least](https://ieeexplore.ieee.org/document/8634878) which makes MIAs less successful. The evaluation of the target models is done with the usual classification metrics.

## Membership inference attack implemented in torch

This section covers a simple implementation of a basic membership inference attack using the `torch` library in Python. The attack is carried in a grey-box setting, meaning that we as adversaries know the architecture of the target model, the hyperparameters used during training, and also the data domain. However, we can only access the final probability distribution outputted by classification models. Our target will be a deep model that classifies handwritten numbers from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). We will train a single shadow model with the same architecture, but on a different dataset although from the same domain. For that, we will use the [EMNIST dataset](http://arxiv.org/abs/1702.05373). Both datasets contain 28x28 pixel images of handwritten symbols. 

For simplicity, we will use a subset of EMNIST that comprises handwritten letters as the non-members when training and testing the attack model. The attack model itself will be a simple fully-connected deep model that takes the output of the SoftMax activation from the classification models as input and returns the probability that the input was a member of their training dataset or not. 

Let's start implementing the attack by importing all necessary libraries. 

```
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import EMNIST, MNIST
from tqdm.notebook import trange
```

Now we can define the architecture of the classification model that we will use for the target model and the shadow model. The models will consist of two parts; the first part uses convolutional layers to extract the most relevant features from the images and the second part applies two dense layers to produce the final prediction. MNIST and the numerical subset of EMNIST include digits from 0 to 9, and therefore the output layer will have 10 nodes. 

```
class ClassifierModel(nn.Module):
    def __init__(self):
        super(ClassifierModel, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=0, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=0, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=0, stride=1),
            nn.LeakyReLU(),
        )
        
        self.stage2 = nn.Sequential(
            nn.Linear(6400, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 10),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        out = self.stage1(x)
        out = out.view(out.size(0), -1)
        out = self.stage2(out)

        return out
```

To train and test the model we need to prepare the data. We will use a single function for that as we need to preprocess the data the same way regardless of the dataset. The type of returned images will be controlled by the parameter `type`. Since this is a simple MIA demonstration, we will limit the number of training and testing images to 2,000. 

```
def get_images(type, batch_size=64):
    split = 2000

    if type is 'mnist':
        train_data = MNIST(
            "mnist", train=True, download=True,
            transform=transforms.ToTensor()
        )
        test_data = MNIST(
            "mnist", train=False, download=True, 
            transform=transforms.ToTensor()
        )
    elif type is 'emnist_digits':
        train_data = EMNIST(
            "emnist", "digits", train=True, download=True, 
            transform=transforms.ToTensor()
        )
        test_data = EMNIST(
            "emnist", "digits", train=False, download=True, 
            transform=transforms.ToTensor()
        )
    elif type is 'emnist_letters':
        train_data = EMNIST(
            "emnist", "letters", train=True, download=True, 
            transform=transforms.ToTensor()
        )
        test_data = EMNIST(
            "emnist", "letters", train=False, download=True, 
            transform=transforms.ToTensor()
        )

    train_data.data = train_data.data[:split]
    train_data.targets = train_data.targets[:split]

    test_data.data = test_data.data[:split]
    test_data.targets = test_data.targets[:split]

    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader
```

Since every model in the implementation is a classifier, we can define a single function for training the models. The only difference is that the attack model is a binary classifier so we will use a different loss function. We will call a testing function at the end of training to evaluate how well the model generalises.

```
def test_classifier(model, test_loader, criterion, attack):
    l = [] # loss
    p = [] # predictions
    model.eval()
    with torch.no_grad():
        for data, targets in test_loader:
            output = model(data)
            loss = criterion(output, targets)

            l.append(loss.item())

            if attack:
                r = output.round()
                r = (r == targets).sum()/data.shape[0]
            else:
                r = output.argmax(dim=1)
                r = (r == targets).sum()/data.shape[0]
            p.append(r.item())

    print(
        'Testing loss: {}, accuracy: {}'.format(
            sum(l)/len(l), sum(p)/len(p)
         )
    )
```

```
def train_classifier(
    model, 
    train_loader, 
    test_loader, 
    epochs=10, 
    attack=False, 
    lr=1e-4
):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if attack:
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in trange(epochs):
        l = [] # loss
        p = [] # predictions
        for data, targets in train_loader:
            opt.zero_grad()
            output = model(data)
            loss = criterion(output, targets)
            loss.backward()
            opt.step()

            l.append(loss.item())

            if attack:
                r = output.round()
                r = (r == targets).sum()/data.shape[0]
            else:
                r = output.argmax(dim=1)
                r = (r == targets).sum()/data.shape[0]
            p.append(r.item())

        print(
            'Epoch: {}, loss: {}, accuracy: {}'.format(
                epoch,sum(l)/len(l),sum(p)/len(p)
            )
        )

    test_classifier(model, test_loader, criterion, attack)

    return model
```

Now we can finally train the target model.

```
mnist_train_loader, mnist_test_loader = get_images(type='mnist')

target_model = ClassifierModel()
target_model = train_classifier(
    target_model, mnist_train_loader, mnist_test_loader, epochs=50
)
```

Similarly, we need to train the shadow model. Its outputs will be used as training data for the attack model.

```
emnist_train_loader,emnist_test_loader = get_images(type='emnist_digits')

shadow_model = ClassifierModel()
shadow_model = train_classifier(
    shadow_model, emnist_train_loader, emnist_test_loader, epochs=50
)
```

When comparing the training and the testing losses the generalisation gap should be reasonably large. That indicates the overconfidence the digit classification models have on the training data. 

We can move on to the adversary. Let's first define the architecture of the attack model. We will use a simple approach with four dense layers. 

```
class AttackNet(nn.Module):
    def __init__(self):
        super(AttackNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(10,500),
            nn.LeakyReLU(),
            nn.Linear(500,500),
            nn.LeakyReLU(),
            nn.Linear(500,500),
            nn.LeakyReLU(),
            nn.Linear(500,1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        return self.layers(x)
```

We also need to define our `torch` dataset class that will return the classification outputs together with the membership label. 

```
class AttackDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
            
        return data, label
```

To prepare the training and testing datasets for the attack model, we need to define a function that gathers the outputs of the digit classification models and also the corresponding membership label. 

```
def get_outputs(model, in_loader, out_loader):
    in_outputs = []
    out_outputs = []
    model.eval() 
    with torch.no_grad():
        for data, _ in in_loader:
            tmp = model(data).detach().cpu().numpy()
            for x in tmp: in_outputs.append(x)

        for data, _ in out_loader:
            tmp = model(data).detach().cpu().numpy()
            for x in tmp: out_outputs.append(x) 

    in_outputs = np.array(in_outputs)
    out_outputs = np.array(out_outputs)

    outputs = torch.FloatTensor(
        np.concatenate((in_outputs, out_outputs))
    ).view((len(in_outputs) + len(out_outputs)),10)

    labels = torch.cat([
        torch.ones((len(in_outputs),1)),
        torch.zeros((len(out_outputs),1))
    ]).view((len(in_outputs) + len(out_outputs)),1)

    return outputs, labels
```

With everything prepared we can finally obtain the data loaders for the attack model.

```
l1_loader, l2_loader = get_images(type='emnist_letters')

victim_outputs, victim_labels = get_outputs(
    victim_model, mnist_train_loader, l1_loader
)
shadow_outputs, shadow_labels = get_outputs(
    shadow_model, emnist_train_loader, l2_loader
)

victim_data = AttackDataset(shadow_outputs, shadow_labels)
shadow_data = AttackDataset(shadow_outputs, shadow_labels)

victim_loader = DataLoader(victim_data, batch_size=64)
shadow_loader = DataLoader(victim_data, batch_size=64)
```

The final step is to initialise the binary classification model and carry on with the attack. 

```
attack_model = AttackNet()
attack_model = train_classifier(
    attack_model, shadow_loader, victim_loader, 
    attack=True, lr=5e-5, epochs=100
)
```

The attack accuracy with this implementation should be around 70% which is a substantial success when it comes to inferring information about membership of training data samples. 

## Conclusion

Security and privacy are important topics in machine learning and with easier access to data and computation power, the vulnerabilities of deep models can be a cause for concern. Membership inference attacks in particular can reveal whether a data entry was included during the training of a model, potentially breaching the privacy of an individual. 