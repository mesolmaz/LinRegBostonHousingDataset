
# coding: utf-8

# # Regression in Python
# 
# ***
# This is a very quick run-through of some basic statistical concepts, adapted from [Lab 4 in Harvard's CS109](https://github.com/cs109/2015lab4) course. Please feel free to try the original lab if you're feeling ambitious :-) The CS109 git repository also has the solutions if you're stuck.
# 
# * Linear Regression Models
# * Prediction using linear regression
# * Some re-sampling methods    
#     * Train-Test splits
#     * Cross Validation
# 
# Linear regression is used to model and predict continuous outcomes while logistic regression is used to model binary outcomes. We'll see some examples of linear regression as well as Train-test splits.
# 
# 
# The packages we'll cover are: `statsmodels`, `seaborn`, and `scikit-learn`. While we don't explicitly teach `statsmodels` and `seaborn` in the Springboard workshop, those are great libraries to know.
# ***

# <img width=600 height=300 src="https://imgs.xkcd.com/comics/sustainable.png"/>
# ***

# In[1]:

# special IPython command to prepare the notebook for matplotlib and other libraries
get_ipython().magic('pylab inline')

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn

import seaborn as sns

# special matplotlib argument for improved plots
from matplotlib import rcParams
sns.set_style("whitegrid")
sns.set_context("poster")


# ***
# # Part 1: Linear Regression
# ### Purpose of linear regression
# ***
# <div class="span5 alert alert-info">
# 
# <p> Given a dataset $X$ and $Y$, linear regression can be used to: </p>
# <ul>
#   <li> Build a <b>predictive model</b> to predict future values of $X_i$ without a $Y$ value.  </li>
#   <li> Model the <b>strength of the relationship</b> between each dependent variable $X_i$ and $Y$</li>
#     <ul>
#       <li> Sometimes not all $X_i$ will have a relationship with $Y$</li>
#       <li> Need to figure out which $X_i$ contributes most information to determine $Y$ </li>
#     </ul>
#    <li>Linear regression is used in so many applications that I won't warrant this with examples. It is in many cases, the first pass prediction algorithm for continuous outcomes. </li>
# </ul>
# </div>
# 
# ### A brief recap (feel free to skip if you don't care about the math)
# ***
# 
# [Linear Regression](http://en.wikipedia.org/wiki/Linear_regression) is a method to model the relationship between a set of independent variables $X$ (also knowns as explanatory variables, features, predictors) and a dependent variable $Y$.  This method assumes the relationship between each predictor $X$ is linearly related to the dependent variable $Y$.  
# 
# $$ Y = \beta_0 + \beta_1 X + \epsilon$$
# 
# where $\epsilon$ is considered as an unobservable random variable that adds noise to the linear relationship. This is the simplest form of linear regression (one variable), we'll call this the simple model. 
# 
# * $\beta_0$ is the intercept of the linear model
# 
# * Multiple linear regression is when you have more than one independent variable
#     * $X_1$, $X_2$, $X_3$, $\ldots$
# 
# $$ Y = \beta_0 + \beta_1 X_1 + \ldots + \beta_p X_p + \epsilon$$ 
# 
# * Back to the simple model. The model in linear regression is the *conditional mean* of $Y$ given the values in $X$ is expressed a linear function.  
# 
# $$ y = f(x) = E(Y | X = x)$$ 
# 
# ![conditional mean](images/conditionalmean.png)
# http://www.learner.org/courses/againstallodds/about/glossary.html
# 
# * The goal is to estimate the coefficients (e.g. $\beta_0$ and $\beta_1$). We represent the estimates of the coefficients with a "hat" on top of the letter.  
# 
# $$ \hat{\beta}_0, \hat{\beta}_1 $$
# 
# * Once you estimate the coefficients $\hat{\beta}_0$ and $\hat{\beta}_1$, you can use these to predict new values of $Y$
# 
# $$\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x_1$$
# 
# 
# * How do you estimate the coefficients? 
#     * There are many ways to fit a linear regression model
#     * The method called **least squares** is one of the most common methods
#     * We will discuss least squares today
#     
# #### Estimating $\hat\beta$: Least squares
# ***
# [Least squares](http://en.wikipedia.org/wiki/Least_squares) is a method that can estimate the coefficients of a linear model by minimizing the difference between the following: 
# 
# $$ S = \sum_{i=1}^N r_i = \sum_{i=1}^N (y_i - (\beta_0 + \beta_1 x_i))^2 $$
# 
# where $N$ is the number of observations.  
# 
# * We will not go into the mathematical details, but the least squares estimates $\hat{\beta}_0$ and $\hat{\beta}_1$ minimize the sum of the squared residuals $r_i = y_i - (\beta_0 + \beta_1 x_i)$ in the model (i.e. makes the difference between the observed $y_i$ and linear model $\beta_0 + \beta_1 x_i$ as small as possible). 
# 
# The solution can be written in compact matrix notation as
# 
# $$\hat\beta =  (X^T X)^{-1}X^T Y$$ 
# 
# We wanted to show you this in case you remember linear algebra, in order for this solution to exist we need $X^T X$ to be invertible. Of course this requires a few extra assumptions, $X$ must be full rank so that $X^T X$ is invertible, etc. **This is important for us because this means that having redundant features in our regression models will lead to poorly fitting (and unstable) models.** We'll see an implementation of this in the extra linear regression example.
# 
# **Note**: The "hat" means it is an estimate of the coefficient.  

# ***
# # Part 2: Boston Housing Data Set
# 
# The [Boston Housing data set](https://archive.ics.uci.edu/ml/datasets/Housing) contains information about the housing values in suburbs of Boston.  This dataset was originally taken from the StatLib library which is maintained at Carnegie Mellon University and is now available on the UCI Machine Learning Repository. 
# 
# 
# ## Load the Boston Housing data set from `sklearn`
# ***
# 
# This data set is available in the [sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html#sklearn.datasets.load_boston) python module which is how we will access it today.  

# In[2]:

from sklearn.datasets import load_boston
boston = load_boston()


# In[3]:

boston.keys()


# In[4]:

boston.data.shape


# In[5]:

# Print column names
print(boston.feature_names)


# In[6]:

# Print description of Boston housing data set
print(boston.DESCR)


# Now let's explore the data set itself. 

# In[7]:

bos = pd.DataFrame(boston.data)
bos.head()


# There are no column names in the DataFrame. Let's add those. 

# In[8]:

bos.columns = boston.feature_names
bos.head()


# Now we have a pandas DataFrame called `bos` containing all the data we want to use to predict Boston Housing prices.  Let's create a variable called `PRICE` which will contain the prices. This information is contained in the `target` data. 

# In[9]:

print(boston.target.shape)


# In[10]:

bos['PRICE'] = boston.target
bos.head()


# ## EDA and Summary Statistics
# ***
# 
# Let's explore this data set.  First we use `describe()` to get basic summary statistics for each of the columns. 

# In[11]:

bos.describe()


# ### Scatter plots
# ***
# 
# Let's look at some scatter plots for three variables: 'CRIM', 'RM' and 'PTRATIO'. 
# 
# What kind of relationship do you see? e.g. positive, negative?  linear? non-linear? 

# In[12]:

plt.figure(figsize=(6,6))
plt.scatter(bos.CRIM, bos.PRICE)
plt.xlabel("Per capita crime rate by town (CRIM)")
plt.ylabel("Housing Price")
plt.title("Relationship between CRIM and Price")


# **Your turn**: Create scatter plots between *RM* and *PRICE*, and *PTRATIO* and *PRICE*. What do you notice? 

# In[13]:

#your turn: scatter plot between *RM* and *PRICE*
plt.figure(figsize=(6,6))
plt.scatter(bos.RM, bos.PRICE)
plt.xlabel("Average number of rooms per dwelling (RM)")
plt.ylabel("Housing Price")
plt.title("Relationship between RM and Price")


# In[14]:

#your turn: scatter plot between *PTRATIO* and *PRICE*
plt.figure(figsize=(6,6))
plt.scatter(bos.PTRATIO, bos.PRICE)
plt.xlabel("pupil-teacher ratio by town (PTRATIO)")
plt.ylabel("Housing Price")
plt.title("Relationship between RM and Price")


# **Your turn**: What are some other numeric variables of interest? Plot scatter plots with these variables and *PRICE*.

# In[15]:

#your turn: create some other scatter plots
f, axarr = plt.subplots(2, 2);
axarr[0, 0].scatter(bos.NOX, bos.PRICE)
axarr[0, 0].set_title('Relationship between NOX and Price')
axarr[0, 0].set_xlabel("nitric oxides concentration (parts per 10 million) (NOX)", fontsize=15)
axarr[0, 0].set_ylabel("Housing Price", fontsize=15)
axarr[0, 1].scatter(bos.AGE, bos.PRICE)
axarr[0, 1].set_title('Relationship between AGE and Price')
axarr[0, 1].set_xlabel("proportion of owner-occupied units built prior to 1940 (AGE)", fontsize=15)
axarr[0, 1].set_ylabel("Housing Price", fontsize=15)
axarr[1, 0].scatter(bos.RAD, bos.PRICE)
axarr[1, 0].set_title('Relationship between RAD and Price')
axarr[1, 0].set_xlabel("index of accessibility to radial highways (RAD)", fontsize=15)
axarr[1, 0].set_ylabel("Housing Price", fontsize=15)
axarr[1, 1].scatter(bos.B, bos.PRICE)
axarr[1, 1].set_title('Relationship between B and Price')
axarr[1, 1].set_xlabel("1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town (B)", fontsize=15)
axarr[1, 1].set_ylabel("Housing Price", fontsize=15)
#f.subplots_adjust(hspace=0.5)
f.tight_layout()
# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
#plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
#plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False);


# I decided to use a subplot to plot 4 different scatter plots. 

# ### Scatter Plots using Seaborn
# ***
# 
# [Seaborn](https://stanford.edu/~mwaskom/software/seaborn/) is a cool Python plotting library built on top of matplotlib. It provides convenient syntax and shortcuts for many common types of plots, along with better-looking defaults.
# 
# We can also use [seaborn regplot](https://stanford.edu/~mwaskom/software/seaborn/tutorial/regression.html#functions-to-draw-linear-regression-models) for the scatterplot above. This provides automatic linear regression fits (useful for data exploration later on). Here's one example below.

# In[16]:

plt.figure(figsize=(7,7))
ax = sns.regplot(y="PRICE", x="RM", data=bos, fit_reg = True)
ax.set(xlabel='Average number of rooms per dwelling (RM)', ylabel='Housing Price')


# ### Histograms
# ***
# 

# Histograms are a useful way to visually summarize the statistical properties of numeric variables. They can give you an idea of the mean and the spread of the variables as well as outliers.

# In[17]:

plt.figure(figsize=(7,7))
plt.hist(bos.CRIM)
plt.title("CRIM")
plt.xlabel("Crime rate per capita")
plt.ylabel("Frequency")
plt.show()


# **Your turn**: Plot separate histograms and one for *RM*, one for *PTRATIO*. Any interesting observations?

# In[18]:

#your turn
plt.figure(figsize=(7,7));
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True);
ax1.hist(bos.RM, 10, normed=1, facecolor='g', alpha=0.75);
#ax1.set_title('Sharing Y axis')
ax2.hist(bos.PTRATIO, 10, normed=1, facecolor='g', alpha=0.75);

ax1.set_xlabel("Average number of rooms per dwelling (RM)", fontsize=18, color='red');
ax1.set_ylabel("Frequency", fontsize=22);
ax2.set_xlabel("pupil-teacher ratio by town (PTRATIO)", fontsize=18, color='red');
ax2.set_ylabel("Frequency", fontsize=22);


# My observations: RM is more tightly distributed while PTRATIO is spread more. RM looks very much like a Normal distribution.

# ## Linear regression with  Boston housing data example
# ***
# 
# Here, 
# 
# $Y$ = boston housing prices (also called "target" data in python)
# 
# and
# 
# $X$ = all the other features (or independent variables)
# 
# which we will use to fit a linear regression model and predict Boston housing prices. We will use the least squares method as the way to estimate the coefficients.  

# We'll use two ways of fitting a linear regression. We recommend the first but the second is also powerful in its features.

# ### Fitting Linear Regression using `statsmodels`
# ***
# [Statsmodels](http://statsmodels.sourceforge.net/) is a great Python library for a lot of basic and inferential statistics. It also provides basic regression functions using an R-like syntax, so it's commonly used by statisticians. While we don't cover statsmodels officially in the Data Science Intensive, it's a good library to have in your toolbox. Here's a quick example of what you could do with it.

# In[19]:

# Import regression modules
# ols - stands for Ordinary least squares, we'll use this
import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[20]:

# statsmodels works nicely with pandas dataframes
# The thing inside the "quotes" is called a formula, a bit on that below
m = ols('PRICE ~ RM',bos).fit()
print(m.summary())


# #### Interpreting coefficients
# 
# There is a ton of information in this output. But we'll concentrate on the coefficient table (middle table). We can interpret the `RM` coefficient (9.1021) by first noticing that the p-value (under `P>|t|`) is so small, basically zero. We can interpret the coefficient as, if we compare two groups of towns, one where the average number of rooms is say $5$ and the other group is the same except that they all have $6$ rooms. For these two groups the average difference in house prices is about $9.1$ (in thousands) so about $\$9,100$ difference. The confidence interval gives us a range of plausible values for this difference, about ($\$8,279, \$9,925$), definitely not chump change. 

# ####  `statsmodels` formulas
# ***
# This formula notation will seem familiar to `R` users, but will take some getting used to for people coming from other languages or are new to statistics.
# 
# The formula gives instruction for a general structure for a regression call. For `statsmodels` (`ols` or `logit`) calls you need to have a Pandas dataframe with column names that you will add to your formula. In the below example you need a pandas data frame that includes the columns named (`Outcome`, `X1`,`X2`, ...), bbut you don't need to build a new dataframe for every regression. Use the same dataframe with all these things in it. The structure is very simple:
# 
# `Outcome ~ X1`
# 
# But of course we want to to be able to handle more complex models, for example multiple regression is doone like this:
# 
# `Outcome ~ X1 + X2 + X3`
# 
# This is the very basic structure but it should be enough to get you through the homework. Things can get much more complex, for a quick run-down of further uses see the `statsmodels` [help page](http://statsmodels.sourceforge.net/devel/example_formulas.html).
# 

# Let's see how our model actually fit our data. We can see below that there is a ceiling effect, we should probably look into that. Also, for large values of $Y$ we get underpredictions, most predictions are below the 45-degree gridlines. 

# **Your turn:** Create a scatterpot between the predicted prices, available in `m.fittedvalues` and the original prices. How does the plot look?

# In[21]:

# your turn
fig, ax = plt.subplots(figsize=(8,6))

ax.scatter(bos.RM, bos.PRICE, label="data")
ax.scatter(bos.RM, m.fittedvalues, label="OLS")
ax.set_xlabel("Average number of rooms per dwelling (RM)")
ax.set_ylabel("Housing Price")
ax.set_title("Relationship between RM and Price")
ax.legend(loc='best');


# ### Fitting Linear Regression using `sklearn`
# 

# In[22]:

from sklearn.linear_model import LinearRegression
X = bos.drop('PRICE', axis = 1)
print(X.shape)
# This creates a LinearRegression object
lm = LinearRegression()
lm


# #### What can you do with a LinearRegression object? 
# ***
# Check out the scikit-learn [docs here](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html). We have listed the main functions here.

# Main functions | Description
# --- | --- 
# `lm.fit()` | Fit a linear model
# `lm.predit()` | Predict Y using the linear model with estimated coefficients
# `lm.score()` | Returns the coefficient of determination (R^2). *A measure of how well observed outcomes are replicated by the model, as the proportion of total variation of outcomes explained by the model*

# #### What output can you get?

# In[23]:

# Look inside lm object

#lm.<tab>


# Output | Description
# --- | --- 
# `lm.coef_` | Estimated coefficients
# `lm.intercept_` | Estimated intercept 

# ### Fit a linear model
# ***
# 
# The `lm.fit()` function estimates the coefficients the linear regression using least squares. 

# In[24]:

# Use all 13 predictors to fit linear regression model
lm.fit(X, bos.PRICE)


# **Your turn:** How would you change the model to not fit an intercept term? Would you recommend not having an intercept?
# 

# ### Estimated intercept and coefficients
# 
# Let's look at the estimated coefficients from the linear model using `1m.intercept_` and `lm.coef_`.  
# 
# After we have fit our linear regression model using the least squares method, we want to see what are the estimates of our coefficients $\beta_0$, $\beta_1$, ..., $\beta_{13}$: 
# 
# $$ \hat{\beta}_0, \hat{\beta}_1, \ldots, \hat{\beta}_{13} $$
# 
# 

# In[25]:

print('Estimated intercept coefficient:', lm.intercept_)


# In[26]:

print('Number of coefficients:', len(lm.coef_))


# In[27]:

# The coefficients
dict(zip(X.columns, lm.coef_), columns = ['features', 'estimatedCoefficients'])


# ###### I replaced 'pd.Dataframe' (suggested in the original exercise) with 'dict', which is suitable for Python 3.

# ### Predict Prices 
# 
# We can calculate the predicted prices ($\hat{Y}_i$) using `lm.predict`. 
# 
# $$ \hat{Y}_i = \hat{\beta}_0 + \hat{\beta}_1 X_1 + \ldots \hat{\beta}_{13} X_{13} $$

# In[28]:

# first five predicted prices
lm.predict(X)[0:5]


# **Your turn:** 
# 
# * Histogram: Plot a histogram of all the predicted prices
# * Scatter Plot: Let's plot the true prices compared to the predicted prices to see they disagree (we did this with `statsmodels` before).

# In[29]:

# your turn
plt.figure(figsize=(7,7))
plt.hist(lm.predict(X))
plt.title("predicted prices")
plt.xlabel("Housing Price")
plt.ylabel("Frequency")
plt.show()


# ### Residual sum of squares
# 
# Let's calculate the residual sum of squares 
# 
# $$ S = \sum_{i=1}^N r_i = \sum_{i=1}^N (y_i - (\beta_0 + \beta_1 x_i))^2 $$

# In[30]:

print(np.sum((bos.PRICE - lm.predict(X)) ** 2))


# #### Mean squared error
# ***
# This is simple the mean of the residual sum of squares.
# 
# **Your turn:** Calculate the mean squared error and print it.

# In[31]:

#your turn
# The mean squared error
print("Mean squared error: %.2f" % np.mean((lm.predict(X) - bos.PRICE) ** 2))


# ## Relationship between `PTRATIO` and housing price
# ***
# 
# Try fitting a linear regression model using only the 'PTRATIO' (pupil-teacher ratio by town)
# 
# Calculate the mean squared error. 
# 

# In[32]:

lm = LinearRegression()
lm.fit(X[['PTRATIO']], bos.PRICE)


# In[33]:

msePTRATIO = np.mean((bos.PRICE - lm.predict(X[['PTRATIO']])) ** 2)
print(msePTRATIO)


# We can also plot the fitted linear regression line. 

# In[34]:

plt.figure(figsize=(7,7))
plt.scatter(bos.PTRATIO, bos.PRICE)
plt.xlabel("Pupil-to-Teacher Ratio (PTRATIO)")
plt.ylabel("Housing Price")
plt.title("Relationship between PTRATIO and Price")

plt.plot(bos.PTRATIO, lm.predict(X[['PTRATIO']]), color='red', linewidth=3)
plt.show()


# # Your turn
# ***
# 
# Try fitting a linear regression model using three independent variables
# 
# 1. 'CRIM' (per capita crime rate by town)
# 2. 'RM' (average number of rooms per dwelling)
# 3. 'PTRATIO' (pupil-teacher ratio by town)
# 
# Calculate the mean squared error. 

# In[35]:

# your turn

lr = LinearRegression()
lr.fit(X[['CRIM', 'RM', 'PTRATIO']], bos.PRICE)
X[['CRIM', 'RM', 'PTRATIO']].shape
#bos.PRICE.shape
mse2 = np.mean((bos.PRICE - lr.predict(X[['CRIM', 'RM', 'PTRATIO']])) ** 2)
print("Mean squared error: %.2f" % mse2)


# 
# ## Other important things to think about when fitting a linear regression model
# ***
# <div class="span5 alert alert-danger">
# <ul>
#   <li>**Linearity**. The dependent variable $Y$ is a linear combination of the regression coefficients and the independent variables $X$. </li>
#   <li>**Constant standard deviation**. The SD of the dependent variable $Y$ should be constant for different values of X.  
#         <ul>
#             <li>e.g. PTRATIO
#         </ul>
#     </li>
#   <li> **Normal distribution for errors**.  The $\epsilon$ term we discussed at the beginning are assumed to be normally distributed. 
#   $$ \epsilon_i \sim N(0, \sigma^2)$$
# Sometimes the distributions of responses $Y$ may not be normally distributed at any given value of $X$.  e.g. skewed positively or negatively. </li>
# <li> **Independent errors**.  The observations are assumed to be obtained independently.
#     <ul>
#         <li>e.g. Observations across time may be correlated
#     </ul>
# </li>
# </ul>  
# 
# </div>
# 

# # Part 3: Training and Test Data sets
# 
# ### Purpose of splitting data into Training/testing sets
# ***
# <div class="span5 alert alert-info">
# 
# <p> Let's stick to the linear regression example: </p>
# <ul>
#   <li> We built our model with the requirement that the model fit the data well. </li>
#   <li> As a side-effect, the model will fit <b>THIS</b> dataset well. What about new data? </li>
#     <ul>
#       <li> We wanted the model for predictions, right?</li>
#     </ul>
#   <li> One simple solution, leave out some data (for <b>testing</b>) and <b>train</b> the model on the rest </li>
#   <li> This also leads directly to the idea of cross-validation, next section. </li>  
# </ul>
# </div>
# 
# ***
# 
# One way of doing this is you can create training and testing data sets manually. 

# In[36]:

X_train = X[:-50]
X_test = X[-50:]
Y_train = bos.PRICE[:-50]
Y_test = bos.PRICE[-50:]
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# Another way, is to split the data into random train and test subsets using the function `train_test_split` in `sklearn.cross_validation`. Here's the [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.train_test_split.html).

# In[37]:

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, bos.PRICE, test_size=0.33, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# **Your turn:**  Let's build a linear regression model using our new training data sets. 
# 
# * Fit a linear regression model to the training set
# * Predict the output on the test set

# In[38]:

# your turn
lr = LinearRegression().fit(X_train, Y_train)
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))
lr.predict(X_test)


# **Your turn:**
# 
# Calculate the mean squared error 
# 
# * using just the test data
# * using just the training data
# 
# Are they pretty similar or very different? What does that mean?

# In[39]:

# your turn
mse3 = np.mean((Y_train - lr.predict(X_train)) ** 2)
print("Mean squared error for Training Data: %.2f" % mse3)

mse4 = np.mean((Y_test - lr.predict(X_test)) ** 2)
print("Mean squared error for Test Data: %.2f" % mse4)


# The MSE values are not too different from each other. We can say thay Train-Test split is well done.

# #### Residual plots

# In[40]:

plt.scatter(lr.predict(X_train), lr.predict(X_train) - Y_train, c='b', s=40, alpha=0.5)
plt.scatter(lr.predict(X_test), lr.predict(X_test) - Y_test, c='g', s=40)
plt.hlines(y = 0, xmin=0, xmax = 50)
plt.title('Residual Plot using training (blue) and test (green) data')
plt.ylabel('Residuals')


# **Your turn:** Do you think this linear regression model generalizes well on the test data?

# In the figure above, there are no extreme points with very high residuals. The spread of the residual is similar to the predicted value's. Also, the spread is largely equal above and below the mean of residual, zero
# 
# This is why the linear regression model generalized well on the test data.

# ### K-fold Cross-validation as an extension of this idea
# ***
# <div class="span5 alert alert-info">
# 
# <p> A simple extension of the Test/train split is called K-fold cross-validation.  </p>
# 
# <p> Here's the procedure:</p>
# <ul>
#   <li> randomly assign your $n$ samples to one of $K$ groups. They'll each have about $n/k$ samples</li>
#   <li> For each group $k$: </li>
#     <ul>
#       <li> Fit the model (e.g. run regression) on all data excluding the $k^{th}$ group</li>
#       <li> Use the model to predict the outcomes in group $k$</li>
#       <li> Calculate your prediction error for each observation in $k^{th}$ group (e.g. $(Y_i - \hat{Y}_i)^2$ for regression, $\mathbb{1}(Y_i = \hat{Y}_i)$ for logistic regression). </li>
#     </ul>
#   <li> Calculate the average prediction error across all samples $Err_{CV} = \frac{1}{n}\sum_{i=1}^n (Y_i - \hat{Y}_i)^2$ </li>
# </ul>
# </div>
# 
# ***
# 
# Luckily you don't have to do this entire process all by hand (``for`` loops, etc.) every single time, ``sci-kit learn`` has a very nice implementation of this, have a look at the [documentation](http://scikit-learn.org/stable/modules/cross_validation.html).

# **Your turn (extra credit):** Implement K-Fold cross-validation using the procedure above and Boston Housing data set using $K=4$. How does the average prediction error compare to the train-test split above?

# In[41]:

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[42]:

X = bos.drop('PRICE', axis = 1)
# Create linear regression object
linreg = LinearRegression()
#.fit(X_train, Y_train)
kfold = KFold(n_splits=4)
scores = cross_val_score(linreg, X, bos.PRICE, cv=kfold)
print("Cross-validation scores: {}".format(scores))
print(scores.mean())


# In[ ]:




# In[ ]:



