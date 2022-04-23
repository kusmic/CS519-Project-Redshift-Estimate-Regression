###############################################################################
#Class to run different regression methods
###############################################################################

import inspect
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#Create a class to run different regression methods
class Methods(object):
   
   def __init__(self, max_trials, min_samples, residual_threshold,
            alpha, solver='auto', random_state=1,xtrain=[],ytrain=[], xtest=[]):
      self.max_trials = max_trials
      self.min_samples = min_samples
      self.residual_threshold = residual_threshold
      self.alpha = alpha
      self.solver = solver
     
      self.random_state = random_state
      self.xtrain = xtrain    
      self.ytrain = ytrain
      self.xtest = xtest
       
      self.__obj = None

   def call(self, method):
      return getattr(self, method)()

   def __fit(self):
     starttime = time()
     self.__obj.fit(self.xtrain, self.ytrain)
     runtime = (time() - starttime)*1000
     print(inspect.stack()[1][3].split("_", 1)[1] + "->" +" The training time is: " 
                + str(runtime)+" ms")

   def __predict(self):
      starttime = time()
      ytrain_pred = self.__obj.predict(self.xtrain)
      runtime = (time() - starttime)*1000
      print(inspect.stack()[1][3].split("_", 1)[1] + "->" 
                                 +" The prediction time for training data is: " 
                    + str(runtime)+" ms")
      starttime = time()
      ytest_pred = self.__obj.predict(self.xtest)
      runtime = time() - starttime
      print(inspect.stack()[1][3].split("_", 1)[1] + "->" + 
                                   " The prediction time for testing data is: " 
                   + str(runtime)+" ms")

      return ytrain_pred, ytest_pred

   def method_linearregression(self):
        self.__obj = LinearRegression()
        self.__fit()
        return self.__predict()

   def method_ransac(self):
        self.__obj = RANSACRegressor(LinearRegression(), min_samples=self.min_samples,
                    residual_threshold=self.residual_threshold, max_trials=self.max_trials,
                    random_state=self.random_state)
        
        self.__fit()
        return self.__predict()

   def method_ridge(self):
        self.__obj = Ridge(alpha=self.alpha, solver=self.solver, 
                                          random_state=self.random_state)
        self.__fit()
        return self.__predict()

   def method_lasso(self):
        self.__obj = Lasso(alpha=self.alpha, random_state=self.random_state)
        self.__fit()
        return self.__predict()


#_____________________________________________________________________________
#The main program:
    
datafile = "mldata_log_est.csv"


#specify the parameters
#**************************************************************
max_trials=300
min_samples=5
residual_threshold=0.1
alpha=1.0
random_state=1
test_size=0.25
solver = 'auto'   


# Loading the data
df = pd.read_csv(datafile)

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Standardizing
sc_x = StandardScaler()
x_std = sc_x.fit_transform(x)

sc_y = StandardScaler()
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()                           
# Splitting train and test data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=test_size, 
                                                     random_state=random_state)

x_std_tr, x_std_ts, y_std_tr, y_std_ts = train_test_split(x_std, y_std, 
                                test_size=test_size, random_state=random_state)

# Apply the algorithm
regression_methods = ["linearregression", "ransac", "ridge", "lasso"]
print("\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/")
for regression_approach in regression_methods:
    regressor = Methods(max_trials=max_trials, min_samples=min_samples, 
                residual_threshold=residual_threshold, alpha=alpha,
                solver=solver, random_state=random_state, xtrain=xtrain, 
                ytrain=ytrain, xtest=xtest)
    
    regressor_std = Methods(max_trials=max_trials, min_samples=min_samples, 
               residual_threshold=residual_threshold, alpha=alpha,
               solver=solver, random_state=random_state, xtrain=x_std_tr, 
               ytrain=y_std_tr, xtest=x_std_ts)
    

    print("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
    print("Before Standardizing:\n")
    ytrain_pred, ytest_pred = regressor.call("method_" + regression_approach)
    #print("---------------------------------------")
    print("\nAfter Standardizing:\n")
    y_std_tr_pred, y_std_ts_pred = regressor_std.call("method_" + regression_approach)
    
    mse_train = mean_squared_error(ytrain, ytrain_pred)
    mse_test = mean_squared_error(ytest, ytest_pred)
    mse_std_train = mean_squared_error(y_std_tr, y_std_tr_pred)
    mse_std_test = mean_squared_error(y_std_ts, y_std_ts_pred)
    
    r2_train = r2_score(ytrain, ytrain_pred)
    r2_test = r2_score(ytest, ytest_pred)
    r2_std_train = r2_score(y_std_tr, y_std_tr_pred)
    r2_std_test = r2_score(y_std_ts, y_std_ts_pred)
    
    print("----------------------------")
    print("\nPerformance Measure : Mean Squared Error:\n")
    print(regression_approach + "-->" + "              MSE train: %.3f, test: %.3f" % (mse_train, mse_test))
    print(regression_approach + "-->" + "          Std MSE train: %.3f, test: %.3f" % (mse_std_train, mse_std_test))
    print("\nPerformance Measure : R^2\n")
    print(regression_approach + "-->" + "              R^2 train: %.3f, test: %.3f" % (r2_train, r2_test))
    print(regression_approach + "-->" + "          Std R^2 train: %.3f, test: %.3f" % (r2_std_train, r2_std_test))
    