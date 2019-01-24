import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error

alldata = np.loadtxt('datasets/CASP.csv', skiprows=1, delimiter=',')

Yall = alldata[:,0]
Xall = alldata[:,1:]

Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xall, Yall, train_size=0.8)

pipeline = make_pipeline(
    #StandardScaler(with_mean=False),
    #DummyRegressor()
    #DecisionTreeRegressor()
    RandomForestRegressor()
    #GradientBoostingRegressor()
    #Ridge()
    #Lasso()
    #LinearRegression()
    #MLPRegressor()

)
#print(cross_validate(pipeline, Xtrain, Ytrain, scoring='neg_mean_squared_error'))

pipeline.fit(Xtrain, Ytrain)
print(mean_squared_error(Ytest, pipeline.predict(Xtest)))