from ucimlrepo import fetch_ucirepo 

import pandas as pd

from sklearn.model_selection import train_test_split

hepatitis = fetch_ucirepo(id=46) 

# data (as pandas dataframes) 
var = hepatitis.data.features 
target = hepatitis.data.targets -1

df = pd.concat([var,target],axis=1)

# Remove Protime due to extreme amo(ut of missing data
df = df.drop(['Protime'], axis=1)

df = df.dropna()

y = df['Class']

X = df.drop(columns=['Class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=44)

