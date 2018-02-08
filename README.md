# Modeling Lung Cancer Incidence in Select U.S. Counties
My Galvanize Capstone Project

## ***Motivation***
Cancer is one of the leading causes of death both in the U.S. and worldwide, and lung cancer is one of the most widespread and deadly varieties. Much research has been done on the link between smoking tobacco and lung cancer risk; it is estimated that ~85% of lung cancer cases are attributed to smoking tobacco. The remaining 10-15% is not as well understood (but a very significant portion given how prevalent lung cancer is). Air pollution, as well as exposure to various chemicals (radon, asbestos, arsenic, cadmium, etc.) explain many of the remaining lung cancer cases.

Recently there has been increased focus on considering the interplay between many different health and environmental factors when determining risk for chronic diseases. For example, a heavy smoker exposed to bad air pollution is significantly more at risk for lung cancer than a heavy smoker not exposed to bad air pollution or a nonsmoker exposed to bad air pollution. By looking at a wide variety of the known variables related to lung cancer, one might be able to better predict the number of to-be diagnosed lung cancer cases per year at a local level (county-wide), providing healthcare professionals and policymakers with actionable insights (catching lung cancer early drastically increases chances of survival).


![](Visuals/county_boxplot3.png)

### Figure 1: Distribution of countywide age and gender standardized lung cancer incidence per 100,000 people between 2001-2011

It is evident that counties differ drastically in their risk for lung cancer - counties in Kentucky show incidence of 150 per 100,00, while counties in California hover around 30 per 100,000, a 5-fold difference.

![](Visuals/state_rates2.png)

### Figure 2: State-wide mean lung cancer incidence per 100,000 people between 2001-2011

Not only do individual counties differ drastically from each other, but counties in different states also display substantial differences in mean lung cancer incidence.


## ***Data Sources***

I requested research access to the NIH SEER Cancer Data, which comprises both cancer incidence and population data for several U.S. states from 1973-2014. I also found public county-wide data on adult smoking levels, radon levels, PM 2.5 levels, ozone levels, toxic releases and air quality index values. For this analysis I limited my time horizon to 2001-2011 due to the best data availability during this period. The full list of data sources used can be found in data_dictionary.txt


### ***Cleaning & Standardization***

A considerable amount of time was spent cleaning and grouping the SEER data so that it could be joined with the other data sources mentioned above.

The only county-wide smoking data I could find were age and gender standardized (according to U.S. census methodology) so that adult smoking percentages can be compared among counties without looking at the role that gender and age play in determining smoking behavior. I decided to use this same methodology to compute age and gender standardized lung cancer incidence figures per 100,000, using age groups <65 and 65+. More detailed explanations of my methodology can be found in methods.txt


## ***Feature Selection***

When deciding which of the features to include in my models, I compared the Bayesian Information Criteria (BIC) and Akaike Information Criteria (AIC) scores of various Lasso regressions that I ran, each including a different set of predictors:

![](Visuals/bic_table5.png)

### Table 1: Comparing feature sets using BIC scores

These estimtors are able to help deal with the overfitting problem mentioned previously. The first component of the BIC, called the likelihood function, is a measure of goodness of fit between a model and the data. The more features you include in your model, the lower your likelihood function will be (the lower the better). The second component of BIC is the regularization parameter. This term penalizes models by the number of features included. So models containing extra features that don't add much information will show higher scores (worse).

The model which minimizes the BIC is comprised of features:
* Adult Daily Smoking % Estimates
* Days of Harmful PM 2.5 Levels
* Air Quality Index Levels
* Mean Radon Levels


## ***Primary Assumptions Behind Linear Regression***

### 1. Sample data representative of population

Here it would be wise to consider what population makes sense. All U.S. Counties? Probably not. The data in this analysis is limited - we only have cancer data on counties in 7 states. Also, the health and environmental data I gathered tends to be more available in larger counties (>100,000 people). Therefore, it would make more sense to say that the relevant population is large U.S. counties.

### 2. True relationship between X and Y is linear

![](Visuals/linear_model.png)

### Figure 3: Mean lung cancer incidence per 100,000 for select counties between 2001-2011

These individual linear best fit lines look pretty good. Linear modeling should produce good results.

Although the least squares lines look like they do a good job of explaining incidence over time in different counties, they are surely overfitting and the predictions generated from such a model would not generalize to other counties/future years.

### 3. Features are Linearly Independent

![](Visuals/heatmap.png)

### Figure 4: Heatmap showing correlations among features and target

There does not seem to be any collinearity between features that we should worry about. One interesting finding, though, is that there is a negative correlation between Median Air Quality index values and cancer incidence in these counties. Daily smoking is clearly the strongest predictor of lung cancer while log radon levels and Days of high PM2.5 seem to be adding some information as well.

### 4. Residuals are Independent and Normally Distributed

![](Visuals/resids_dist.png)
![](Visuals/probplot.png)

### 5. Variance of residuals is constant (homoscedasticity)

![](Visuals/resid_varplot.png)

## ***Simple Linear Regression***

Without introducing a hierarchical struture to the data, we have 3 options:
1. Fully-Pooled: Model 2001-2011 lung cancer incidence through use of a single regression model for all counties
2. State-Pooled: Model 2001-2011 lung cancer incidence through use of separate regression models for each state
3. Unpooled: Model 2001-2011 lung cancer incidence by running separate regression models on each individual county


I tried both fully-pooled and unpooled models, and chose to evaluate model performance on Root Mean Square Error (RMSE), a measure of the standard deviation of model predictions from actual values:

![](Visuals/RMSE.png)

A lower RMSE value is desired. The fully-pooled model had an RMSE of 18.6, and the unpooled an RMSE of 10.3. Relative to the mean lung cancer incidence of ~70 for all counties, the unpooled model wasn't bad. But clearly the fully-pooled model is not a good option.

![](Visuals/predictions3.png)

### Figure 4: Unpooled estimates vs. actual mean lung cancer incidence per county

This plot shows that the unpooled model does a very good job of estimating the mean incidence per county. But how good is it at generalizing to future years or to other counties? Probably not great. Both counties and states share many similarites that would explain lung cancer incidence that I have not included in my model, such as smoking prevention initiatives and air quality standards. These confounding variables could be very useful when forecasting county-wide incidence, but the unpooled model does not take them into account. Therefore, in order to improve upon these baseline models, I chose to focus on multilevel regression which helps control for confounding variables.

## ***Multilevel Regression***

Multilevel, or hierarchical regression techniques are a compromise between the pooled and unpooled methods. This approach assumes that while the coefficients (y-intercept, slope terms) are different for each county, they all come from a common group distribution ("prior").

![](Visuals/multilevel_formula.png)

This type of parameter estimation is core to Bayesian Statistics. While
frequentist methods assume that model coefficients are always fixed, Bayesian methods try to estimate the coefficients using sampling techniques such as the Markov Chain Monte Carlo algorithm (which I used).

I tried 2 multilevel models, differing by the group distributions specified. The first used state-level grouping, so that the prior distribution for each county is made up of all other counties in that state. The second grouped all counties together to create a prior distribution.


Comparing RMSE between the two models:
1. State-Level: 13.0
2. County-Level: 7.7

Overall I obtained the best results by grouping all counties together to form a prior distribution. Lets take a look at Warren County, KY to get a better sense for the difference between these 2 models:

![](Visuals/hier_counties13.png)

Kentucky counties have the highest lung cancer incidence out of all states in my data. The statewide average is ~100 per 100,000. Even though Warren County seems to show significantly lower incidence ~80-85, the dark green line is shifted upwards towards the group mean. The dark blue line, however, fits the local data very well and is closer to the overall mean for counties in my dataset ~70 per 100,000.

Looking at plots from a few other counties:


It is clear from these plots that the County-Level model produces point estimates and 95% confidence intervals that fit the data much better than the state-grouped approach, evident by the model's significantly lower RMSE.

Looking at these point estimates and 95% confidence intervals across all counties:

![](Visuals/multilevel_comparison.png)

### Figure 5: Point estimates and 95% Confidence Intervals for Multilevel Models

### ***Caveat***

Since I was limited by data availability for many of the variables I used, there are some states with very few counties in my dataset. For example, I only have data on 3 counties in Michigan, even though there are 83 in the state. Using the state-level hierarchical model, counties in these states are highly influenced by only a few other counties in the same state, which may be why this model underperforms the baseline unpooled model.

## ***Future Direction***

To address the issue above, my next step would be to combine the strengths of both the state-level and county-level hierarchical models. I could try only using the state-level models on counties where there is sufficient data on other counties in the same state.

I would also like to try analyzing other data that is related to lung cancer incidence (socioeconomic, other health factors).

In addition to standardizing the data by age and gender, it would be helpful to standardize by race/ethnicity as well, since this is another confounding variable that explains differences in incidence.

Lastly, in order to build a strong predictive model that could generalize well to other counties/years, it would be necessary to obtain unaggregated healthcare data with many more data points per county.
