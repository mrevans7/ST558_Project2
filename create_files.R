library(tidyverse)
library(knitr)
library(caret)
library(gbm)
library(randomForest)
library(rmarkdown)

#Read Data
news <- read_csv("OnlineNewsPopularity.csv")

#Add column that tells which weekday it is
news <- news %>%
  mutate(Weekday = case_when(weekday_is_monday ==  1 ~ 'Monday',
                             weekday_is_tuesday ==  1 ~ 'Tuesday',
                             weekday_is_wednesday ==  1 ~ 'Wednesday',
                             weekday_is_thursday ==  1 ~ 'Thursday',
                             weekday_is_friday ==  1 ~ 'Friday',
                             weekday_is_saturday ==  1 ~ 'Saturday',
                             weekday_is_sunday ==  1 ~ 'Sunday'))

#get unique teams
weekdays <- unique(news$Weekday)

#create filenames
output_file <- paste0(weekdays, ".md")

#create a list for each team with just the team name parameter
params_list = lapply(weekdays, FUN = function(x){list(weekday = x)})

#put into a data frame
reports <- tibble(output_file, params_list)

reports

pwalk(reports, render, input = "README.Rmd")

params_test <- list(
  weekday = "Monday",
  weekday = "Tuesday"
)

test <- function(day){
  rmarkdown::render(
  "README.Rmd", 
  params = list(weekday = day),
  output_file = paste0(day,"-", "Report", ".md")
)
}

test_days <- c("Monday", "Tuesday", "Wednesday")

lapply(test_days ,test)

