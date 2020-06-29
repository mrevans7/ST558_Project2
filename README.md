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

#Split into Training and Testing Set
set.seed(321)
train <- sample(1:nrow(daily_news), size = nrow(daily_news) * 0.7)
test <- setdiff(1:nrow(daily_news), train)

#Subset data
daily_train <- daily_news[train, ]
daily_test <- daily_news[test, ]
```

## Purpose of Analysis

## Methods

# Summarizations

# Modeling

# Conclusion
