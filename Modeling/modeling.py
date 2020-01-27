import pickle, os, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pymc3 as pm
from pymc3 import Model, sample, Normal, HalfCauchy, Uniform, forestplot
from scipy.stats import boxcox, probplot, norm
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet



class LinearModel(object):
    def __init__(self, data, features, target, type='linear',test_size=0.2):
        '''
        INPUT:
            - data: Target and features to be modeled
            - features: Independent variables to be modeled
            - target: Target variable name
            - test_size: Percent of test cases
            - type: Linear model type (linear, lasso, elastic net)
        OUTPUT: Instantiated class
        '''
        # Construct dataframe with features to be included in model
        self.data = data
        # Define the independent variables
        self.features = features
        # Scale features
        self.scale_features()
        # Define the target variable
        self.target = self.data[target].values
        # Train/test size
        self.test_size = test_size
        # Train test split
        self.split()

        if type=='linear':
            self.fit_predict_linear()
        elif type=='lasso':
            self.fit_predict_lasso()
        else:
            self.fit_predict_net()

        self.rmse()

    def scale_features(self):
        scaler = preprocessing.StandardScaler()
        self.scaled_features = scaler.fit_transform(self.data[self.features])

    def split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.scaled_features, self.target, test_size=self.test_size, random_state = 17)

    def fit_predict_linear(self):
        lm = LinearRegression()
        lm.fit(self.X_train, self.y_train)
        self.predictions = lm.predict(self.X_test)

    def fit_predict_lasso(self):
        lasso = Lasso(alpha=0.1)
        lasso.fit(self.X_train, self.y_train)
        self.predictions = lasso.predict(self.X_test)
        self.lasso_variables = np.array(list(zip(self.features,lasso.coef_)))

    def fit_predict_net(self):
        net = ElasticNet(alpha=0.1, l1_ratio=1)
        net.fit(self.X_train, self.y_train)
        self.predictions = net.predict(self.X_test)

    def rmse(self):
        self.rmse =  np.sqrt(mean_squared_error(self.y_test, self.predictions))


class BayesianModel(object):
    def __init__(self, data, features, target, group, train_year = 2010, progressbar = True):
        '''
        INPUT:
            - data: Target and features to be modeled
            - features: Independent variables to be modeled
            - target: Target variable name
            - group: Group to pool model (state-pooled, unpooled)
            - train_year: Year cutoff for data to be used in building model
            - progressbar: Display progressbar when running model?
        OUTPUT: Instantiated class
        '''
        # Construct dataframe with features to be included in model
        self.data = data
        # Define the independent variables
        self.features = features
        # Scale features
        self.scale_features()
        # Define the target variable
        self.target = self.data[target].values
        # Define the years to be used for training
        self.train_year = train_year
        # Specified hierarchy
        self.group = group
        # Prepare model given specified pooling
        self.determine_pooling()
        # Get a list of hyperprior names
        self.hyperprior_names, self.prior_names = self.get_prior_names()
        # Get model trace
        self.trace, self.model = self.build_model()
        # Get model estimates, standard errors and residuals
        self.estimates, self.standard_errors, self.residuals = self.calc_results()
        # Get model score (rmse)
        self.rmse = self.rmse()

    def scale_features(self):
        '''
        INPUT: None
        OUTPUT: Scaled features as numpy array
        '''
        scaler = preprocessing.StandardScaler()
        self.scaled_features = scaler.fit_transform(self.data[self.features])


    def determine_pooling(self):
        if self.group == 'states':
            col = 'State'
        else:
            col = 'State_and_county'
        self.uniques = self.data[col].unique()
        self.lookup = dict(zip(self.uniques, range(len(self.uniques))))
        self.entity = self.data[col].replace(self.lookup)

    def get_prior_names(self):
        '''
        INPUT: None
        OUTPUT: Creates a list of coefficient names to be passed to Pymc3 model for evaluation
        '''
        # Get a list of hyperprior names
        hyperprior_names = list(zip(['mu_beta_' + x for x in self.features], ['sigma_beta_' + x for x in self.features]))
        # Get a list of coefficient names
        prior_names = ['beta_' + x for x in self.features]
        return hyperprior_names, prior_names

    def build_model(self):
        with Model() as bayesian_model:
            # Prior for intercept term
            a = pm.Normal('alpha', mu=0., sd=1e5, shape=len(self.data))
            sigma = HalfCauchy('sigma', 5)
            # Priors for coefficients
            coef_list = []
            for i, coef in enumerate(self.features):
                coef_list.append(Normal('beta_'+self.features[i], mu=0., sd=1e5))

            # Expected value of lung cancer
            cancer_est = a[self.entity]
            for i, x in enumerate(coef_list):
                cancer_est += x * self.scaled_features[:,i]

            # Data likelihood
            y_like = Normal('y_like', mu=cancer_est, sd=sigma, observed=self.target)


        with bayesian_model:
            # hierarchical_trace = pm.sample(1000, n_init=150000, tune=100000)
            bayesian_trace = sample(100, n_init=1500, tune=1000)

        return bayesian_trace, bayesian_model


    def calc_results(self):
        est = self.trace['alpha'].mean(axis=0)[self.entity]
        std = self.trace['alpha'].std(axis=0)[self.entity]
        for i, coef in enumerate(self.prior_names):
            est += self.trace['beta_'+self.features[i]].mean(axis=0)*self.scaled_features[:,i]
            std += self.trace['beta_'+self.features[i]].std(axis=0)*self.scaled_features[:,i]
        resids = self.target-est
        return est, std, resids


    def rmse(self):
        rmse = np.sqrt(mean_squared_error(self.target,self.estimates))
        return rmse

class HierarchicalModel(BayesianModel):
    def __init__(self, data, features, target, group, train_year = 2010, progressbar = True):
        # Construct dataframe with features to be included in model
        self.data = data
        # Define the independent variables
        self.features = features
        # Scale features
        BayesianModel.scale_features(self)
        # Define the target variable
        self.target = self.data[target].values
        # Define the years to be used for training
        self.train_year = train_year
        # Specified hierarchy
        self.group = group
        # Prepare model given specified pooling
        BayesianModel.determine_pooling(self)
        # Get a list of hyperprior names
        self.hyperprior_names, self.prior_names = BayesianModel.get_prior_names(self)
        # Get model trace
        self.trace, self.model = self.build_hierarchical_model()
        # Get model estimates, standard errors and residuals
        self.estimates, self.standard_errors, self.residuals = self.calc_results()
        # Get model score (rmse)
        self.rmse = BayesianModel.rmse(self)



    def build_hierarchical_model(self):
        with pm.Model() as hierarchical_model:
            # Hyperpriors for intercept mean, standard deviation
            mu_a = pm.Normal('mu_alpha', mu=0., sd=1e5)
            sigma_a = pm.HalfCauchy('sigma_alpha', beta=5)

            # Coefficient hyperpriors
            coef_hyperpriors = []
            for i, theta in enumerate(self.features):
                mu = pm.Normal(self.hyperprior_names[i][0], mu=0., sd=1e5)
                sigma = pm.HalfCauchy(self.hyperprior_names[i][1], beta=5)
                coef_hyperpriors.append((mu, sigma))


            # Prior for intercept term (not hierarchical - separate for each county)
            a = pm.Normal('alpha', mu=mu_a, sd=sigma_a, shape=len(self.data))

            # Priors for coefficients
            coef_list = []
            for i, coef in enumerate(coef_hyperpriors):
                coef_list.append(pm.Normal('beta_'+self.features[i], mu=coef[0], sd=coef[1], shape=len(self.uniques)))

            # Model error
            eps = pm.HalfCauchy('eps', beta=1)

            # Expected value of lung cancer
            cancer_est = a
            for i, x in enumerate(coef_list):
                cancer_est += x[self.entity] * self.scaled_features[:,i]

            # Data likelihood
            y_like = pm.Normal('y_like', mu=cancer_est, sd=eps, observed=self.target)


        with hierarchical_model:
            # hierarchical_trace = pm.sample(1000, n_init=150000, tune=100000)
            hierarchical_trace = pm.sample(100, n_init=1500, tune=1000)

        return hierarchical_trace, hierarchical_model

    def calc_results(self):
        est = self.trace['alpha'].mean(axis=0)
        std = self.trace['alpha'].std(axis=0)
        for i, coef in enumerate(self.prior_names):
            est += self.trace['beta_'+self.features[i]].mean(axis=0)[self.entity]*self.scaled_features[:,i]
            std += self.trace['beta_'+self.features[i]].std(axis=0)[self.entity]*self.scaled_features[:,i]
        resids = self.target-est
        return est, std, resids

#### Plotting functions ####
def dist_plot(residuals, x_label='Distribution of Model Residuals',color='blue',font_size=12, label_size=10):
    fig, ax = plt.subplots()
    ax = sns.distplot(residuals, fit=norm, color=color)
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.tick_params(labelsize=label_size)

def resid_plot(estimates,y,color='red'):
    fig = plt.subplot()
    sns.residplot(estimates, y, lowess=True, color=color, line_kws={'color':'black'})

def prob_plot(residuals):
    fig = plt.subplot()
    probplot(residuals, plot=plt)


def unpooled_plot(y_new,c):
    county_cancer = y_new.loc[c]['Cancer_Rate'].values
    year_range = range(y_new.Year.min(),y_new.Year.max()+1)

    plt.grid(False)
    plt.scatter([str(i) for i in year_range],county_cancer, color='black', s=200)

    X = np.array([int(i) for i in year_range])
    Y = np.array(county_cancer)
    fit = np.polyfit(X, Y, deg=1)
    plt.plot(range(len(X)), fit[0] * X + fit[1], color='black', lw=5)

    plt.scatter([str(i) for i in year_range], y_new.loc[c]['Unpooled_Estimates'].values, color='red', s=200)

    plt.xlabel('Years', fontsize=15)
    plt.ylabel('Mean Lung Cancer Incidence per 100,000', fontsize=15)
    plt.title(c, fontsize=15)
    plt.tick_params(labelsize=12)
    plt.ylim([20,140])
    plt.tight_layout()

def multi_unpooled_plot(y_new,counties,size):
    fig, ax = plt.subplots(figsize=size)
    counter = 1
    num = len(counties)
    for c in counties:
        if np.mod(num,2)==0:
            plt.subplot(num/2,num/2,counter)
        else:
            plt.subplot((num//2)+1,num/2,counter)
        unpooled_plot(y_new,c)
        counter += 1


def county_plot(y_new,c):
    county_cancer = y_new.loc[c]['Cancer_Rate'].values
    year_range = range(y_new.Year.min(),y_new.Year.max()+1)

    plt.grid(False)
    plt.plot([str(i) for i in year_range],county_cancer, '.k-', lw=3, markersize=15)
    X = np.array([int(i) for i in year_range])
    Y = np.array(county_cancer)

    county_estimates = y_new.loc[c]['County_Estimates'].values
    county_se = y_new.loc[c]['County_Standard_Error'].values

    z = np.polyfit(X, county_estimates, deg=1)
    plt.plot([str(i) for i in year_range], county_estimates, '.b-', lw=3, markersize=15)

    plt.fill_between(range(len(X)), z[0] * X + z[1] + county_se,z[0] * X + z[1]-county_se, interpolate=True, color = 'blue', alpha = 0.4, label = '95%_CI')

    #COUNTY MEAN LINE
    county_mean = y_new['Cancer_Rate'].mean()
    plt.axhline(y=county_mean, color='#00FFFF', ls='--', lw=3)

    state_estimates = y_new.loc[c]['State_Estimates'].values
    state_se = y_new.loc[c]['State_Standard_Error'].values

    s = np.polyfit(X, state_estimates, deg=1)
    plt.plot([str(i) for i in year_range], state_estimates, '.g-', lw=3, markersize=15)

    plt.fill_between(range(len(X)), s[0] * X + s[1]+ state_se, s[0] * X + s[1]-1*state_se, interpolate=True, color= 'green', alpha = 0.4, label = '95%_CI')

    ## STATE MEAN LINE
    state_mean = y_new[y_new.State==c[-2:]]['Cancer_Rate'].mean()
    plt.axhline(y=state_mean, color='#7CFC00', ls='--', lw=3)

    plt.xlabel('Years', fontsize=15)
    plt.ylabel('Lung Cancer Incidence per 100,000', fontsize=15)
    plt.title(c, fontsize=15)
    plt.tick_params(labelsize=12)
    plt.ylim([20,140])

    plt.tight_layout()


def multi_county_plot(y_new,counties,size):
    fig, ax = plt.subplots(figsize=size)
    counter = 1
    num = len(counties)
    for c in four_counties:
        if np.mod(num,2)==0:
            plt.subplot(num/2,num/2,counter)
        else:
            plt.subplot((num//2)+1,num/2,counter)
        county_plot(y_new,c)
        counter += 1

def forest_plot(y_new, estimates):
    for estimate, color in estimates.items():
        for i, m, se in zip(range(len(y_new.index.unique())), y_new.groupby(by='State_and_county')[estimate+'_Estimates'].mean().sort_values(ascending=True), y_new.groupby(by='State_and_county')[estimate+'_Standard_Error'].mean().sort_values(ascending=True)):
            plt.plot([i,i], [m-se, m+se], color+'-', lw=1)


        plt.scatter(range(len(y_new.index.unique())), y_new.groupby(by='State_and_county')[estimate+'_Estimates'].mean().sort_values(ascending=True),color=color,s=10)
    plt.ylim(0,200)
    plt.xticks([], [])
    plt.tick_params(labelsize=10)
    plt.ylabel('Mean Cancer Incidence 95% Confidence Intervals', fontsize=10)
    plt.xlabel('Ordered Counties', fontsize=10)
    plt.title('Mean County Estimates from Hierarchical Regressions', fontsize=12)
    plt.tight_layout()



def create_target_df(df, unpooled_estimates, county_estimates, county_se, state_estimates, state_se):
    y_new = pd.DataFrame({
            'Cancer_Rate':df['Cancer_Rate'],
            'Year':df['Year'],
            'Count':[i for i in range(len(df))],
            'State':df['State'],
            'Unpooled_Estimates': unpooled_estimates,
            'County_Estimates': county_estimates,
            'County_Standard_Error':county_se,
            'State_Estimates': state_estimates,
            'State_Standard_Error':state_se
    })

    y_new.index = df['State_and_county']
    return y_new



if __name__=='__main__':
    filepath = '/home/dhense/PublicData/ZNAHealth/intermediate_files/'
    engineered_pickle = 'engineered.pickle'

    print("...loading pickle")
    tmp = open(filepath+engineered_pickle,'rb')
    df = pickle.load(tmp)
    tmp.close()

    features = ['Smoking_daily', 'Median_AQI', 'Radon_mean']
    target = 'Cancer_Rate'
    group = 'counties'

    # model = LinearModel(df, features, target, type='lasso', test_size=0.2)
    # print(model.rmse)

    unpooled = {'pickle':'unpooled_trace.pickle'}
    hier_state = {'pickle':'hier_state_trace.pickle'}
    hier_county = {'pickle':'hier_county_trace.pickle'}

    # Select model(s) to run/load
    pickles = [unpooled, hier_state, hier_county]

    for p in pickles:
        if not os.path.isfile(filepath+p['pickle']):
            if p['pickle']=='unpooled_trace.pickle':
                p['model'] = BayesianModel(df, features, target, group='counties')
                p['trace'] = p['model'].trace
            elif p['pickle']=='hier_state_trace.pickle':
                p['model'] = HierarchicalModel(df, features, target, group='states')
                p['trace'] = p['model'].trace
            else:
                p['model'] = HierarchicalModel(df, features, target, group='counties')
                p['trace'] = p['model'].trace

            print("...saving pickle")
            tmp = open(filepath+p['pickle'],'wb')
            pickle.dump((p['model'],p['trace']),tmp)
            tmp.close()

        else:
            print("...loading pickle")
            tmp = open(filepath+p['pickle'],'rb')
            p['model'],p['trace'] = pickle.load(tmp)
            tmp.close()

    print(unpooled['model'].rmse, hier_state['model'].rmse, hier_county['model'].rmse)

    y_new = create_target_df(df, unpooled['model'].estimates, hier_county['model'].estimates, hier_county['model'].standard_errors, hier_state['model'].estimates, hier_state['model'].standard_errors)

    print(y_new)

    '''
    fig, ax = plt.subplots(figsize=(15,10))
    unpooled_plot(y_new,'Whatcom County, WA')
    plt.show()

    fig, ax = plt.subplots(figsize=(15,10))
    county_plot(y_new,'Whatcom County, WA')
    plt.show()

    four_counties = ['Yolo County, CA','Warren County, KY', 'Wayne County, MI', 'Grant County, NM']
    multi_unpooled_plot(y_new, four_counties, (20,10))
    plt.show()
    '''
    fig, ax = plt.subplots(figsize=(15,5))
    estimates = {'County':'b','State':'g'}
    forest_plot(y_new,estimates)
    plt.show()


    # m = HierarchicalModel(df, features, target, group='counties')
    # print(m.rmse)

    # dist_plot(m.residuals)
    # plt.show()
