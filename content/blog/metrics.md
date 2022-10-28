---
title: "Introduction to machine learning metrics"
date: 2022-07-01T15:31:00+02:00
draft: false
---

When evaluating the performance of a machine learning model, the experts must carefully choose the appropriate metrics to determine how effectively their model works. The metrics, unlike the loss functions, are **not employed to optimize the model during training** as they only indicate how well the model performs. Selecting the correct metric can disclose shortcomings and weaknesses in the model which might not be exposed by only assessing the loss function.

Each machine learning task requires a different type of metric. While there are many metrics to choose from, we will only cover the most prevalent ones.

## Classification

Classification is one of the [most common tasks](https://link.springer.com/article/10.1007/s10462-007-9052-3) in machine learning. Typically, the loss function optimised during training is cross-entropy, which provides little intuition to the untrained eye about how well the model performs.

### Accuracy 

For the reasons stated above the evaluation is conventionally done with the accuracy metric. Accuracy is the ratio of correctly classified samples out of the entire test set.

```
y_true = [0, 1, 2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 1, 2, 2, 0, 2]

# Out of 8 predictions, 5 are correct
# Accuracy ~= 0.635
```

**Top-k accuracy** is a slightly different metric that considers a prediction valid if the true class is among the top *k* predictions ranked by their scores. One of the use cases of this metric is measuring the performance of a search engine or a recommendation system. Assuming the top returned items are similar (since they all cover the same relevant information) it might be difficult for the model to distinguish which items should be ranked higher or lower. By measuring only the top-1 accuracy, i.e., the simple accuracy from above, the metric might indicate low performance, while top-5 accuracy might convey a completely different outcome. For *k* larger than 1, the metric requires the prediction scores for each class. This is because it needs to rank them and find the top *k* predictions.

```
y_true = [0, 1, 2]

y_pred = [
	[0.3, 0.6, 0.1],
   	[0.2, 0.1, 0.8],
	[0.1, 0.5, 0.2],
]

# Top-1 accuracy  = 0.0
# Top-2 accuracy ~= 0.67
# Top-3 accuracy  = 1.0
```

### F1-score

Precision and recall are two common metrics used together in [binary classification](https://ieeexplore.ieee.org/abstract/document/5340935). It is important to be familiar with the following concepts in order to understand how they are derived.

* True positive: sample labelled with 1 classified as 1
* False positive: sample labelled with 0 classified as 1
* True negative: sample labelled with 0 classified as 0
* False negative: sample labelled with 1 classified as 0

**Precision** is the ratio between the number of correctly classified positive cases (i.e., true positives) and the total number of cases classified as positive (correctly and incorrectly, i.e., true positives and false positives). 

```
precision = TP/(TP + FP)
```

**Recall** is the ratio between the number of correctly classified positive samples (i.e., true positives) and the number of total positive samples in the testing set (i.e., true positives and false negatives). 

```
recall = TP/(TP + FN)
```

Precision and recall can be combined into the **F1-score** as follows.

```
function f_score(precision, recall):
    n = precision * recall
    d = precision + recall
    return 2 * n / d
```

The F1-score shines in cases where the classes are imbalanced. Assume a set of data points with patients' biomarkers labelled with 0 or 1 depending on whether the patient suffers from a particular medical condition or not. The data can be used to build a classifier that can predict whether a patient is likely to develop the condition or not. If the training dataset is imbalanced, it will lead to [overfitting](https://pubs.acs.org/doi/full/10.1021/ci0342472) resulting in high accuracy values, while the utility of such a model will be low. Let's see below why is that possible. 

```
y_true = [0, 0, 0, 0, 0, 1, 1, 0, 0, 1]
y_pred = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

accuracy  = 0.8
f1_score  = 0.5
```

Although the accuracy is 80%, the model was able to recognise only one positive patient out of three leading to low recall. For comparison, we can assume a second model that doesn't suffer from overfitting. 

```
y_true = [0, 0, 0, 0, 0, 1, 1, 0, 0, 1]
y_pred = [0, 1, 0, 0, 0, 0, 1, 0, 0, 1]

accuracy   = 0.8
f1_score  ~= 0.67
```

While the accuracy is the same, the second model achieves a higher F1-score as out of the three positive patients it can correctly classify two of them. The crucial observation is that we cannot depend only on accuracy. The F1-score is an important statistical measure that can reveal additional information about the performance of a classifier.

### ROC and AUC

Another metric commonly used in data science is the receiver operating characteristic curve, frequently shortened as **ROC**. It is a graphical representation illustrating the performance of a binary classifier given the [varying decision threshold](https://www.sciencedirect.com/science/article/abs/pii/S0001299878800142) used for classifying a sample as positive (usually the threshold is set to 0.5, but it can vary). ROC is created using the following rates. 

* True positive rate (= recall): proportion of correct predictions given positive test cases only
* False positive rate: proportion of in-correct predictions given negative test cases only

```
# true class labels
y_true = [0, 1, 1, 1, 0]

# predicted probabilities for class 1
y_pred = [0.25, 0.71, 0.41, 0.93, 0.12]


Threshold | TPR | FPR 
----------------------
      0.0 | 1.0 | 1.0
      0.2 | 1.0 | 0.5
      0.5 | 0.7 | 0.0
      0.8 | 0.3 | 0.0
      1.0 | 0.0 | 0.0
```

ROC is plotted on a graph, where the x-axis represents the false positive rate and the y-axis represents the true positive rate. The ideal classifier has the true positive rate equal to 1.0 (all positive cases predicted as positive) and the false positive rate equal to 0.0 (all negative cases predicted as negative). When comparing multiple models and their utility the best model is the one with the ROC [curve closest to the top-left corner](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5082211/) of the unit square. The closer the curve is to the corner, the higher similarity to the ideal classifier. An example of ROC curve (independent of the calculation above) is shown below.

![ROC example](/roc.png)

ROC curves are used to calculate the area under the receiver operating characteristic curve, often shortened as **AUC**. While ROC provides a visual representation of the performance, AUC yields a [quantitative evaluation](https://ieeexplore.ieee.org/abstract/document/1388242). Intuitively, the area under the curve will be larger if the true positive rate stays high even with decreasing false positive rate (the area will be equal to 1.0 for the ideal classifier). The [simple interpretation](https://www.math.ucdavis.edu/~saito/data/roc/fawcett-roc.pdf) of the AUC value is that it indicates the probability of the model being able to correctly distinguish between a randomly selected positive and negative sample from the test set. 

## Regression

Regression tasks focus on predicting continuous values based on the input data. For this reason, the evaluation metrics quantify how close the predictions are to the ground truth data and how well they describe the behaviour of the observations.

### Mean errors

The most commonly used loss functions in regression problems are **mean absolute error** (MAE) and **mean squared error** (MSE). Both errors calculate how well the model fits the data based on the average residuals (the differences between the predicted values and the true values). The former takes the absolute residuals while the latter squares them. Deciding on which loss function/metric is better depends on the task itself. Usually, if the data includes outliers far from the predicted values, MSE will penalise them more harshly than MAE due to its quadratic nature. Therefore, when used during optimisation it might [reduce the overall utility](https://ieeexplore.ieee.org/abstract/document/9167435) of the fitted model. The rule of thumb is to use MSE when the target variables follow a Gaussian distribution and MAE when the data includes outliers.

### The coefficient of determination

Measuring the goodness-of-fit of a linear regression model is frequently done through the coefficient of determination, also called **R-squared**. The coefficient expresses how well the independent variables in the regression model explain the variability of the dependent variable. (The independent variables are those used to obtain the dependent variable, i.e., the prediction.) R-squared is computed on a fitted regression model using the following quantities.

* The residual sum of squares: the sum of squared differences between the true observations and the predicted values
* The total sum of squares: the sum of squared differences between the true observations and their mean

The following pseudocode describes the calculation of the coefficient.

```
function RSS(y_true, y_pred):
    s = 0
    for y_i, f_i in y_true, y_pred:
    	s += (y_i - f_i)**2
    return s
    
function TSS(y_true):
    s = 0
    for y_i in y_true:
        s += (y_i - mean(y_true))**2
    return s
    
function R_squared(y_true, y_pred):
    res = RSS(y_true, y_pred)
    tot = TSS(y_true)
    return 1 - res/tot
```

The coefficient of determination in the majority of cases ranges from 0 to 1, where the value 0 means the independent variables cannot explain the variability of the dependent variable at all, and the value 1 indicates that the independent variables perfectly explain the variability of the dependent variable.

However, the coefficient can reach negative values as well. When that happens it means that the fitted model performs even worse than a horizontal line. 

## Computer vision

In computer vision tasks the evaluation of how well an image looks compared to a reference image is most frequently done with the [structural similarity index measure](https://ieeexplore.ieee.org/document/1284395) and the [peak signal-to-noise ratio](https://ieeexplore.ieee.org/abstract/document/5596999).

### Structural similarity index measure

While MAE and MSE can be used to measure the pixel value similarity of two images, they don't consider the structural information of images. To quantify the visual characteristics and spatial similarity, the structural similarity index measure (SSIM) combines luminance, contrast and structure information. SSIM ranges from 0 to 1, where 1 can be obtained only if the two compared images are identical. 

The following pseudocode represents the calculation of SSIM given a reconstructed image `x` and the reference image `y`. The variables `c1, c2, c3` are positive constants to avoid a null denominator.

```
function luminance(x, y):
    x_avg = mean(x)
    y_avg = mean(y)
    
    n = 2 * x_avg * y_avg + c1
    d = x_avg**2 + y_avg**2 + c1
    
    return n/d 
    
function contrast(x, y):
    x_var = variance(x)
    y_var = variance(y)
    
    n = 2 * sqrt(x_var) * sqrt(y_var) + c2
    d = x_var + y_var + c2
    
    return n/d 
    
function structure(x, y):
    xy_cov = covariance(x, y)
    x_var = variance(x)
    y_var = variance(y)
    
    n = xy_cov + c3
    d = x_var * y_var + c3
    
    return n/d
    
function ssim(x, y):
    l = luminance(x, y)
    c = contrast(x, y)
    s = structure(x, y)
    
    return l * c * s
```

Usually, the index is calculated on arbitrarily large sliding windows rather than on full-sized images as it has been shown that it is better to apply SSIM [locally rather than globally](https://ieeexplore.ieee.org/document/1284395) (similarly to convolution). The global value is then calculated as the average of the local indices.

Demonstration of the usefulness of SSIM is shown below. While the middle and the right images have the same MSE compared to the reference image on the left, their image quality is vastly different, which is measured by the SSIM.

![SSIM example](/ssim.png)

### Peak signal-to-noise ratio

The quality of image reconstruction is often measured with the peak signal-to-noise ratio (PSNR). The metric expresses the ratio between the maximum possible value of a signal and the severity of image degradation. In pseudocode, PSNR can be calculated as follows, where `MAX` is the maximum possible pixel value of the image (for example, for 8bit images it is 255).

```
function psnr(x, y):
    mse = MSE(x, y)
    return 20 * log10(MAX/sqrt(mse))
```

## Conclusion 

Evaluating the performance of a machine learning model is essential and should be a part of every solution pipeline. Whether the task is classification, regression, or image reconstruction, it is necessary to know how the model is performing. The metrics can indicate the utility of the model and the steps required to achieve even better results.
