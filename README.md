ST558 Project 2
================
Michael Evans
7/3/2020

  - [Code to Render File for Each Day of the
    Week](#code-to-render-file-for-each-day-of-the-week)
  - [Analysis for Each Day of the
    Week](#analysis-for-each-day-of-the-week)

# Code to Render File for Each Day of the Week

``` r
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
lapply(days, render_function)
```

# Analysis for Each Day of the Week

The analysis for [Monday is available here](Monday-Report.md).  
**Best Model:** Bagged Trees

The analysis for [Tuesday is available here](Tuesday-Report.md).  
**Best Model:** Logistic Regression

The analysis for [Wednesday is available here](Wednesday-Report.md).  
**Best Model:** Logistic Regression

The analysis for [Thursday is available here](Thursday-Report.md).  
**Best Model:** Bagged Trees

The analysis for [Friday is available here](Friday-Report.md).  
**Best Model:** Bagged Trees

The analysis for [Saturday is available here](Saturday-Report.md).  
**Best Model:** Bagged Trees

The analysis for [Sunday is available here](Sunday-Report.md).  
**Best Model:** Bagged Trees

The .Rmd file that was [automated is available here.](Project_Work.Rmd).
