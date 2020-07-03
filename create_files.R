#Load Libraries
library(tidyverse)
library(rmarkdown)

#Create Function to Render File
render_function <- function(day){
  rmarkdown::render(
  "Project_Work.Rmd", 
  params = list(weekday = day),
  output_file = paste0(day,"-", "Report", ".md")
)
}

#Create Vector of Days of the Week
days <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

#Apply Function to Each Day of the Week
lapply(days ,render_function)
