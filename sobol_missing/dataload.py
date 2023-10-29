import numpy as np
import pandas as pd 
import random

import seaborn as sns
import matplotlib.pyplot as plt

#n = 1000
#sim = 1000
#a = 1
#b = 1
#p = 0.5

class dataset:
    def __init__(self,n,p,df):
        #Number of observations
        self.n = n
        #MCAR Bernoulli prob.
        self.p = p
        #data set 
        self.df = df 
        
    #Ishigami function
        #np.random.seed(24)   
    
    def ishigami(self):
        return lambda a, b:  np.sin(self.df['X1']) + a * np.sin(self.df['X2'])**2 + b * (self.df['X3'])**4 * np.sin(self.df['X1']) + np.random.normal(0, 0.05, self.n)
    
    def toyreg(self):
        return lambda b1,b2: b1*self.df['X1'] + b2*self.df['X2'] + np.random.normal(0, 0.1, self.n)
    
    #H-Sample with target from Ishigami function
    
    def h_df(self,a,b):
        self.df['Y'] = self.ishigami()(a,b)
        return self.df['Y']
    
    def picked(self,request,a,b):
        #request are the variables to be randomized, e.g. ['X2','X3']
        newdf = self.df.copy()
        values  = [list(pd.Series(np.random.uniform(0,1,self.n)))]*len(request) #np.random.normal for toy example
        dictionary = dict(zip(request, values))
        for col, new_values in dictionary.items():
            newdf = newdf.assign(**{col: new_values})
        return newdf
    
        
        
    
    def mcar(self,a,b):
        #self.df['Y'] = self.ishigami()(a,b)
        mask = np.random.binomial(size=(self.n,3), n=1, p= self.p) == 1
        mask = np.reshape(mask,(self.n,3))
        #print(mask)
        df1 = self.df.where(mask,np.nan)
        #print(df1)
        return df1.dropna()
    
    def mar(self,cond1,cond2,var1='X2',var2='X1'):
        cond_x = np.where((self.df[var1] <0.7) , 0, 1) # & or (self.df['X3'] <0.1)
        omega = (np.random.binomial(size=(self.n,1), n=1, p= self.p) == 1)  #omega.shape 
        omega = np.reshape(omega,(self.n,))
        condition = (omega & (cond_x  ==1)) | (cond_x  ==0)
        self.df[var2] = np.where(condition,np.array(self.df[var2]),np.nan)
        return self.df.dropna()
    
    def mnar(self,cond1,cond2,var='X1'):
        cond_x = np.where((self.df[var] <0.7) , 1, 0) # & or (self.df['X3'] <0.1)
        omega = (np.random.binomial(size=(self.n,1), n=1, p= self.p) == 1)  #omega.shape 
        omega = np.reshape(omega,(self.n,))
        condition = (omega & (cond_x  ==1)) | (cond_x  ==0)
        self.df[var] = np.where(condition,np.array(self.df[var]),np.nan)
        return self.df.dropna()
    


