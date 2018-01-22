# Predicting Lung Cancer Rates in Select U.S. Counties
My Galvanize Capstone Project

Motivation:
Cancer is one of the leading causes of death both in the U.S. and worldwide, and lung cancer is one of the most widespread and deadly varieties. Much research has been done on the link between smoking tobacco and lung cancer risk; it is estimated that ~85% of lung cancer cases are attributed to smoking tobacco. The remaining 10-15% is not as well understood (but a very significant portion given how prevalent lung cancer is). Air pollution, as well as exposure to various chemicals (radon, asbestos, arsenic, cadmium, etc.) explain many of the remaining lung cancer cases.

Recently there has been increased focus on considering the interplay between many different health and environmental factors when determining risk for chronic diseases. For example, a heavy smoker exposed to bad air pollution is significantly more at risk for lung cancer than a heavy smoker not exposed to bad air pollution or a nonsmoker exposed to bad air pollution. By looking at a wide variety of the known variables related to lung cancer, one might be able to better predict the number of to-be diagnosed lung cancer cases per year at a local level (county-wide), providing healthcare professionals and policymakers with actionable insights (catching lung cancer early drastically increases chances of survival).


![](counties3.png)

### Figure 1: Distribution of countywide age and gender standardized lung cancer incidence per 100,00 between 2000-2014

It is evident that counties differ drastically in their risk for lung cancer - counties in Kentucky show incidence of 120 per 100,00, while counties in Iowa hover around 20 per 100,000, a 6x difference. 

![](state_years2.png)

### Figure 2: State-wide mean lung cancer incidence per 100,000 between 2000-2014

Not only do individual counties differ drastically from each other, but states also display substantial differences in mean lung cancer incidence. 


## Basic Linear Regression
Without introducing a hierarchical struture to the data, we have 3 options:
1. Fully-Pooled: Forecast individual county 2014 lung cancer incidence through use of a single regression model for all counties
2. State-Pooled: Forecast individual county 2014 lung cancer incidence through use of regression models for each state
3. Unpooled: Forecast individual county 2014 lung cancer incidence by running separate regression on each individual county


Given how different individual counties and states look from each other, option 1 does not seem like it would produce good results. 


![](4counties4.png)

### Figure 3: Mean lung cancer incidence per 100,000 for select counties between 2000-2014

These individual linear regression plots look pretty good - 3/4 of the counties display a linear trend in mean cancer rates. 

However, the least squares fit lines do seem to overfit the county-level data by predicting more extreme results for 2014 than we see (King County, Wayne).

Since states have distinct and different distribtions of mean lung cancer rates, it would make sense to improve upon this basic regression model by creating a hierarchical structure by which individual county regressions are constrained by a distribution of paramters (y-intercept and slope) which would result in regression lines that are less extreme and less likely to overfit. 

One step beyond this would be to assume that the states we are looking at are a random sample from the U.S., and that the states themselves could be constrained by U.S.-level hyperparameters. 

## Data Sources

Other than the SEER cancer data used in these simple models, I have gathered county-wide data on adult smoking levels, radon levels, PM 2.5 levels, ozone levels, toxic releases, and air quality index values. The full list of data sources can be found in data_dictionary.txt  
