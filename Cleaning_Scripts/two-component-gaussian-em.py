#!/usr/bin/env python
"""
This is an implementation of two-component Gaussian example from

Elements of Statistical Learning (pp 272)

"""


## make imports
from __future__ import division
import sys
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as stats
import os, pickle
import pandas as pd

class TwoComponentGaussian():

    def __init__(self, y, num_iters, num_runs,verbose=False):
        self.y = y
        self.verbose = verbose
        self.max_like, self.best_est = self.run_em_algorithm(num_iters, num_runs)

    ### make defs for initial guessing, expectation, and maximization
    def get_init_guesses(self,y):

        ## make intial guesses for the parameters (mu1, sig1, mu2, sig2 and pi)
        n    = len(self.y)
        mu1  = y[np.random.randint(0,n)]
        mu2  = y[np.random.randint(0,n)]
        sig1 = np.random.uniform(0.5,1.5)
        sig2 = np.random.uniform(0.5,1.5)
        pi   = 0.5

        return {'n':n, 'mu1':mu1, 'mu2':mu2, 'sig1':sig1, 'sig2':sig2, 'pi':pi}

    def perform_expectation(self, y, parms):
        gamma_hat = np.zeros((parms['n']),'float')

        for i in range(parms['n']):
            phi_theta1 = stats.norm.pdf(y[i],loc=parms['mu1'],scale=np.sqrt(parms['sig1']))
            phi_theta2 = stats.norm.pdf(y[i],loc=parms['mu2'],scale=np.sqrt(parms['sig2']))
            numer = parms['pi'] * phi_theta2
            denom = ((1.0 - parms['pi']) * phi_theta1) + (parms['pi'] * phi_theta2)
            gamma_hat[i] = numer / denom

        return gamma_hat

    def perform_maximization(self,y,parms,gamma_hat):
        """
        maximization
        """

        ## use weighted maximum likelihood fits to get updated parameter estimates
        numer_muhat1 = 0
        denom_hat1 = 0
        numer_sighat1 = 0
        numer_muhat2 = 0
        denom_hat2 = 0
        numer_sighat2 = 0
        pi_hat = 0

        ## get numerators and denomanators for updating of parameter estimates
        for i in range(parms['n']):
            numer_muhat1 = numer_muhat1 + ((1.0 - gamma_hat[i]) * y[i])
            numer_sighat1 = numer_sighat1 + ( (1.0 - gamma_hat[i]) * ( y[i] - parms['mu1'] )**2 )
            denom_hat1 = denom_hat1 + (1.0 - gamma_hat[i])

            numer_muhat2 = numer_muhat2 + (gamma_hat[i] * y[i])
            numer_sighat2 = numer_sighat2 + (gamma_hat[i] * ( y[i] - parms['mu2'] )**2)
            denom_hat2 = denom_hat2 + gamma_hat[i]
            pi_hat = pi_hat + (gamma_hat[i] / parms['n'])

        ## calculate estimates
        mu_hat1 = numer_muhat1 / denom_hat1
        sig_hat1 = numer_sighat1 / denom_hat1
        mu_hat2 = numer_muhat2 / denom_hat2
        sig_hat2 = numer_sighat2 / denom_hat2

        return {'mu1':mu_hat1, 'mu2':mu_hat2, 'sig1': sig_hat1, 'sig2':sig_hat2, 'pi':pi_hat, 'n':parms['n']}

    def get_likelihood(self,y,parms,gamma_hat):
        """
        likelihood
        """
        part1 = 0
        part2 = 0

        for i in range(parms['n']):
            phi_theta1 = stats.norm.pdf(y[i],loc=parms['mu1'],scale=np.sqrt(parms['sig1']))
            phi_theta2 = stats.norm.pdf(y[i],loc=parms['mu2'],scale=np.sqrt(parms['sig2']))
            part1 = part1 + ( (1.0 - gamma_hat[i]) * np.log(phi_theta1) + gamma_hat[i] * np.log(phi_theta2) )
            part2 = part2 + ( (1.0 - gamma_hat[i]) * np.log(parms['pi']) + gamma_hat[i] * np.log(1.0 - parms['pi']) )

        return part1 + part2


    def run_em_algorithm(self, num_iters, num_runs, verbose = True):
        """
        main algorithm functions
        """

        max_like = -np.inf
        best_estimates = None

        for j in range(num_runs):
            iter_count = 0
            parms = self.get_init_guesses(self.y)

            ## iterate between E-step and M-step
            while iter_count < num_iters:
                iter_count += 1

                ## ensure we have reasonable estimates
                if parms['sig1'] < 0.0 or parms['sig2'] < 0.0:
                    iter_count = 1
                    parms = get_init_guesses()

                ## E-step
                gamma_hat = self.perform_expectation(self.y,parms)
                log_like = self.get_likelihood(self.y,parms,gamma_hat)

                ## M-step
                parms = self.perform_maximization(self.y,parms,gamma_hat)

            if log_like > max_like:
                max_like = log_like
                best_estimates = parms.copy()

            if self.verbose == True:
                print('run:',j+1, '--- mu1: ',round(parms['mu1'],2),'--- mu2:',round(parms['mu2'],2),)
                print('--- obs.data likelihood: ', round(log_like,4))

        print("runs complete")


        return max_like, best_estimates

if __name__ == '__main__':

    # y1 = np.array([-0.39,0.12,0.94,1.67,1.76,2.44,3.72,4.28,4.92,5.53])
    # y2 = np.array([ 0.06,0.48,1.01,1.68,1.80,3.25,4.12,4.60,5.28,6.22])
    # y  = np.hstack((y1,y2))
    df = pd.read_csv('lung_dataframe_overall3.csv',converters={'Combined': lambda x: str(x),'State-county recode_x': lambda x: str(x)})
    y = np.array(df.iloc[:,-1:]).flatten()

    num_iters = 25
    num_runs = 20
    verbose = True
    make_plots = True
    tcg = TwoComponentGaussian(y, num_iters, num_runs,verbose=verbose)

    print('max likelihood', tcg.max_like)
    print('best estimates', tcg.best_est)

    if make_plots:
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        n, bins, patches = ax.hist(y,bins=20,facecolor="#9999FF",alpha=0.7,normed=1,histtype='stepfilled')
        #n, bins, patches = plt.hist(y,15,normed=1,facecolor='gray',alpha=0.75)

        ## add a 'best fit' line (book results)
        mu1 = 4.62
        mu2 = 1.06
        sig1 = 0.87
        sig2 = 0.77

        p1 = mlab.normpdf( bins, mu1, np.sqrt(sig1))
        p2 = mlab.normpdf( bins, mu2, np.sqrt(sig2))
        l1 = ax.plot(bins, p1, 'r--', linewidth=1)
        l2 = ax.plot(bins, p2, 'r--', linewidth=1)

        ## add a 'best fit' line (results from here)
        p3 = mlab.normpdf( bins, tcg.best_est['mu1'], np.sqrt(tcg.best_est['sig1']))
        p4 = mlab.normpdf( bins, tcg.best_est['mu2'], np.sqrt(tcg.best_est['sig2']))
        l3 = ax.plot(bins, p3, 'k-', linewidth=1)
        l4 = ax.plot(bins, p4, 'k-', linewidth=1)

        plt.xlabel('y')
        plt.ylabel('freq')
        plt.ylim([0,0.8])

        plt.legend( (l1[0], l3[0]), ('Book Estimate', 'EM Estimate') )

        plt.savefig('../TwoComponentGauss.png')
        plt.show()
