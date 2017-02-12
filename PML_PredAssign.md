---
title: "Practical Machine Learning- Prediction Assignment"
author: "Sagar Shethia"
date: "February 12, 2017"
output: html_document
---

## Background  
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.  

In this project, we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise.  

## Required libraries
```{r, cache = T}
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(corrplot)
```
## Data
The training data for this project are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

### Read the Data
After downloading the data from the data source, we can read the two csv files into two data frames.  
```{r, cache = T}
train <- read.csv("pml-training.csv")
test <- read.csv("pml-testing.csv")
dim(train)
dim(test)
```
The training data set contains 19622 observations and 160 variables, while the testing data set contains 20 observations and 160 variables. The "classe" variable in the training set is the outcome to predict. 

### Cleaning the data
In this step, we will remove missing values as well as some low variance variables.
```{r, cache = T}
sum(complete.cases(train))
```
First, we remove columns that contain NA missing values.
```{r, cache = T}
train <- train[, colSums(is.na(train)) == 0] 
test <- test[, colSums(is.na(test)) == 0] 
```  
Next, we remove some columns that do not contribute much to the accelerometer measurements.
```{r, cache = T}
classe <- train$classe
trainRem <- grepl("^X|timestamp|window", names(train))
train <- train[, !trainRem]
training <- train[, sapply(train, is.numeric)]
training$classe <- classe
testRem <- grepl("^X|timestamp|window", names(test))
test <- test[, !testRem]
testing <- test[, sapply(test, is.numeric)]
```
Now, the cleaned training data set contains 19622 observations and 53 variables, while the testing data set contains 20 observations and 53 variables. The "classe" variable is still in the cleaned training set.

### Partitioning the data
Then, we can split the cleaned training set into a pure training data set (70%) and a validation data set (30%). We will use the validation data set to conduct cross validation in future steps.  
```{r, cache = T}
set.seed(22) # For reproducibile purpose
inTrain <- createDataPartition(training$classe, p=0.70, list=F)
trainData <- training[inTrain, ]
testData <- training[-inTrain, ]
```

## Modeling the Data
We fit a predictive model for activity recognition using **Random Forest** algorithm because it automatically selects important variables and is robust to correlated covariates & outliers in general. We will use **3-fold cross validation** when applying the algorithm.  
```{r, cache = T}
controlRf <- trainControl(method="cv", 3)
model <- train(classe ~ ., data=trainData, method="rf", trControl=controlRf, ntree=25)
model
```
Then, we estimate the performance of the model on the validation data set.  
```{r, cache = T}
predict <- predict(model, testData)
confusionMatrix(testData$classe, predict)
```
```{r, cache = T}
accuracy <- postResample(predict, testData$classe)
accuracy
err <- 1 - as.numeric(confusionMatrix(testData$classe, predict)$overall[1])
err
```
So, the estimated accuracy of the model is 99.42% and the estimated out-of-sample error is 0.58%.

## Predictions for Test Data
Now, we apply the model to the original testing data set downloaded from the data source. We remove the `problem_id` column first.  
```{r, cache = T}
predValidation <- predict(model, testing)
ValidationPredictionResults <- data.frame(
  problem_id=testing$problem_id,
  predicted=predValidation)
print(ValidationPredictionResults)
```  

## Appendix: Figures
1. Correlation Matrix Visualization  
```{r, cache = T}
corrPlot <- cor(trainData[, -length(names(trainData))])
corrplot(corrPlot, method="color")
```
2. Decision Tree Visualization
```{r, cache = T}
treeModel <- rpart(classe ~ ., data=trainData, method="class")
prp(treeModel) # fast plot
```