# Neural Network Charity Analysis

## Overview

With our knowledge of machine learning and neural networks, we’ll use the features in the provided dataset to help create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, we received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

EIN and NAME — Identification columns
APPLICATION_TYPE — Alphabet Soup application type
AFFILIATION — Affiliated sector of industry
CLASSIFICATION — Government organization classification
USE_CASE — Use case for funding
ORGANIZATION — Organization type
STATUS — Active status
INCOME_AMT — Income classification
SPECIAL_CONSIDERATIONS — Special consideration for application
ASK_AMT — Funding amount requested
IS_SUCCESSFUL — Was the money used effectively

## Resources

- Data Source: [charity_data.csv](https://github.com/jasonmontes94/Neural_Network_Charity_Analysis/blob/main/Resources/charity_data.csv)
- Software: Python 3.7.7, Anaconda Navigator 1.9.12, Conda 4.8.4, Jupyter Notebook 6.0.3

## Results


### Data Preprocessing

- The columns EIN and NAME are identification information and have been removed from the input data.
- The column IS_SUCCESSFUL contains binary data refering to weither or not the charity donation was used effectively. This variable is then considered as the target for our deep learning neural network.
- The following columns APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT are the features for our model. Encoding of the categorical variables, spliting into training and testing datasets and standardization have been applied to the features.

### Compiling, Training, and Evaluating the Model
- This deep-learning neural network model is made of two hidden layers with 80 and 30 neurons respectively.
- The input data has 43 features and 25,724 samples.
- The output layer is made of a unique neuron as it is a binary classification.
- To speed up the training process, we are using the activation function ReLU for the hidden layers. As our output is a binary classification, Sigmoid is used on the output layer.
- For the compilation, the optimizer is adam and the loss function is binary_crossentropy.
- The model accuracy is under 75%. This is not a satisfying performance to help predict the outcome of the charity donations.
- To increase the performance of the model, we applied bucketing to the feature ASK_AMT and organized the different values by intervals.
- We increased the number of neurons on one of the hidden layers, then we used a model with three hidden layers.
- We also tried a different activation function (tanh) but none of these steps helped improve the model's performance.
    
## Summary

Attempts to optimize the original model were unsuccessful. Through iterating different parameters of the deep learning model, the highest model accuracy acheived was 72.7%. We could use a supervised machine learning model such as the Random Forest Classifier to combine a multitude of decision trees to generate a classified output and evaluate its performance against our deep learning model.