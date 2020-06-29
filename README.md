ST558 Project 2
================
Michael Evans
7/3/2020

  - [Introduction](#introduction)
      - [Description of Data](#description-of-data)
          - [Reading Data](#reading-data)
      - [Purpose of Analysis](#purpose-of-analysis)
      - [Methods](#methods)
  - [Summarizations](#summarizations)
  - [Modeling](#modeling)
  - [Conclusion](#conclusion)

# Introduction

## Description of Data

### Reading Data

``` r
#Read Data
news <- read_csv("OnlineNewsPopularity.csv")

#Get News for One Day
daily_news <- news %>% filter(weekday_is_monday == 1) %>% select(-(weekday_is_monday:weekday_is_sunday))

#Make Shares into Factor
daily_news$shares_1400 <- ifelse(daily_news$shares > 1400, 1, 0)
daily_news$shares_1400 <- as.factor(daily_news$shares_1400)

#Select Variables
daily_news <-daily_news %>% select(n_tokens_title:num_videos, 
                                   data_channel_is_lifestyle:data_channel_is_world, 
                                   shares_1400)

#Make Factor
daily_news$data_channel_is_lifestyle <- as.factor(daily_news$data_channel_is_lifestyle)
daily_news$data_channel_is_entertainment <- as.factor(daily_news$data_channel_is_entertainment)
daily_news$data_channel_is_bus <- as.factor(daily_news$data_channel_is_bus)
daily_news$data_channel_is_socmed <- as.factor(daily_news$data_channel_is_socmed)
daily_news$data_channel_is_tech <- as.factor(daily_news$data_channel_is_tech)
daily_news$data_channel_is_world <- as.factor(daily_news$data_channel_is_world)

#Split into Training and Testing Set
set.seed(321)
train <- sample(1:nrow(daily_news), size = nrow(daily_news) * 0.7)
test <- setdiff(1:nrow(daily_news), train)

#Subset data
daily_train <- daily_news[train, ]
daily_test <- daily_news[test, ]
```

``` r
#Create Model
library(randomForest)

random_forest <- randomForest(shares_1400 ~ ., data = daily_train, mtry = ncol(daily_train)/3, 
                                            ntree = 500, importance = TRUE)
random_forest
```

    ## 
    ## Call:
    ##  randomForest(formula = shares_1400 ~ ., data = daily_train, mtry = ncol(daily_train)/3,      ntree = 500, importance = TRUE) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 5
    ## 
    ##         OOB estimate of  error rate: 39.47%
    ## Confusion matrix:
    ##      0    1 class.error
    ## 0 1655  825   0.3326613
    ## 1 1015 1167   0.4651696

``` r
#Predict Random Forest
random_pred <- predict(random_forest, daily_test, type = "class")

#Create Results
random_results <- confusionMatrix(data = random_pred, reference = daily_test$shares_1400)
random_results
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 718 453
    ##          1 323 505
    ##                                         
    ##                Accuracy : 0.6118        
    ##                  95% CI : (0.59, 0.6332)
    ##     No Information Rate : 0.5208        
    ##     P-Value [Acc > NIR] : < 2.2e-16     
    ##                                         
    ##                   Kappa : 0.218         
    ##                                         
    ##  Mcnemar's Test P-Value : 3.642e-06     
    ##                                         
    ##             Sensitivity : 0.6897        
    ##             Specificity : 0.5271        
    ##          Pos Pred Value : 0.6132        
    ##          Neg Pred Value : 0.6099        
    ##              Prevalence : 0.5208        
    ##          Detection Rate : 0.3592        
    ##    Detection Prevalence : 0.5858        
    ##       Balanced Accuracy : 0.6084        
    ##                                         
    ##        'Positive' Class : 0             
    ## 

``` r
#View Model
random_forest
```

    ## 
    ## Call:
    ##  randomForest(formula = shares_1400 ~ ., data = daily_train, mtry = ncol(daily_train)/3,      ntree = 500, importance = TRUE) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 5
    ## 
    ##         OOB estimate of  error rate: 39.47%
    ## Confusion matrix:
    ##      0    1 class.error
    ## 0 1655  825   0.3326613
    ## 1 1015 1167   0.4651696

``` r
#Create Model
bagged_tree <- train(shares_1400 ~ ., data = daily_train, method = "treebag", 
                     trControl = trainControl(method = "cv"), tuneLength = 10)

#View Model
bagged_tree
```

    ## Bagged CART 
    ## 
    ## 4662 samples
    ##   15 predictor
    ##    2 classes: '0', '1' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 4196, 4196, 4196, 4195, 4196, 4195, ... 
    ## Resampling results:
    ## 
    ##   Accuracy   Kappa    
    ##   0.5922402  0.1781322

``` r
#Predict Bagged Tree
bagged_pred <- predict(bagged_tree, daily_test)

#Create Results
bagged_results <- confusionMatrix(data = bagged_pred, reference = daily_test$shares_1400)
bagged_results
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 695 456
    ##          1 346 502
    ##                                           
    ##                Accuracy : 0.5988          
    ##                  95% CI : (0.5769, 0.6204)
    ##     No Information Rate : 0.5208          
    ##     P-Value [Acc > NIR] : 1.367e-12       
    ##                                           
    ##                   Kappa : 0.1925          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.0001186       
    ##                                           
    ##             Sensitivity : 0.6676          
    ##             Specificity : 0.5240          
    ##          Pos Pred Value : 0.6038          
    ##          Neg Pred Value : 0.5920          
    ##              Prevalence : 0.5208          
    ##          Detection Rate : 0.3477          
    ##    Detection Prevalence : 0.5758          
    ##       Balanced Accuracy : 0.5958          
    ##                                           
    ##        'Positive' Class : 0               
    ## 

\`\`\`

## Purpose of Analysis

## Methods

# Summarizations

# Modeling

# Conclusion
