import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, ExpSineSquared, Kernel, WhiteKernel, Matern, RBF, RationalQuadratic, Hyperparameter
from sklearn.preprocessing import StandardScaler

# ----------------------------------- Get data ------------------------------------#
stock1 = pd.read_csv(".\stock_data\Starbucks_1.csv",header=None)
stock2 = pd.read_csv(".\stock_data\Adobe_2.csv",header=None)
stock3 = pd.read_csv(".\stock_data\HP_3.csv",header=None)
# column vector
stock1 = np.array(stock1)
stock2 = np.array(stock2)
stock3 = np.array(stock3)

# --------------------------------- Preprocess Data --------------------------------#
# [num_stock,num_training_days,num_predict_days]
# stock_param = [3,50,10]   # short time series
stock_param = [3,400,87]   # long time series

# Standardlization
# ref: https://blog.csdn.net/wzyaiwl/article/details/90549391
Myscaler = StandardScaler()
stock1_std = Myscaler.fit_transform(stock1)
stock2_std = Myscaler.fit_transform(stock2)
stock3_std = Myscaler.fit_transform(stock3)
stock_std = np.concatenate((stock1_std, stock2_std, stock3_std),axis=1)

# Assign training data
stock_train = stock_std[:stock_param[1],:]   # all stocks
stock_train_pls1 = stock_std[:stock_param[1]+1,:]   # all stocks
x_train = np.arange(stock_param[1]).reshape(-1,1)
x_train_pls1 = np.arange(stock_param[1]+1).reshape(-1,1)

# ------------------------------- Fit Model and Predict ---------------------------#
MyKernel = (
    # final kernel of long time series

    DotProduct(sigma_0=5) +
    0.1*WhiteKernel(noise_level=0.01) +
    0.1*ExpSineSquared(length_scale=10,periodicity=10)
    # 0.1*RBF(length_scale=10)
    # 0.1*Matern(length_scale=10,nu=1.5)
)

MyGPR = GaussianProcessRegressor(kernel = MyKernel, n_restarts_optimizer=10, random_state=0)

x_new = np.arange(stock_param[1]+stock_param[2]).reshape(-1,1)

# first stock
MyGPR.fit(x_train, stock_train[:,0])
stock_predict1, sigma1= MyGPR.predict(x_new, return_std=True)

# second stock
MyGPR.fit(x_train, stock_train[:,1])
stock_predict2, sigma2= MyGPR.predict(x_new, return_std=True)

# third stock
MyGPR.fit(x_train, stock_train[:,2])
stock_predict3, sigma3= MyGPR.predict(x_new, return_std=True)

stock_predict1.reshape(-1,1)
stock_predict2.reshape(-1,1)
stock_predict3.reshape(-1,1)

x_predict_real = np.arange(stock_param[1],stock_param[1]+stock_param[2]).reshape(-1,1)

# ---------------------------------- Virtualization -------------------------------#
wastes, Myfigs = plt.subplots(3, 1, figsize=(8, 8))

# stock1
Myfigs[0].plot(x_train_pls1,stock_train_pls1[:,0],"r")
Myfigs[0].plot(x_new,stock_predict1,"b")
Myfigs[0].plot(x_predict_real,stock1_std[stock_param[1]:stock_param[1]+stock_param[2],:],"k")
upper_bound1 = stock_predict1 + 1.96*sigma1
lower_bound1 = stock_predict1 - 1.96*sigma1
Myfigs[0].fill_between(np.squeeze(x_new),lower_bound1,upper_bound1,alpha=0.5)
Myfigs[0].set_xlabel("Time(day)")
Myfigs[0].set_ylabel("Target Price(standardlized)")
Myfigs[0].set_title("StarBucks")
# Myfigs[0].set_title(
#     "Kernel & Hyperparameters: %s\n"
#     % (MyGPR.kernel_)
# )
Myfigs[0].set_xlim((0,stock_param[1]+stock_param[2]-1))
Myfigs[0].grid(True)

# # stock1:single kernel fuction test
# pylab.grid(True)
# plt.plot(x_train_pls1,stock_train_pls1[:,0],"r")
# plt.plot(x_new,stock_predict1,"b")
# plt.plot(x_predict_real,stock1_std[stock_param[1]:stock_param[1]+stock_param[2],:],"k")
# upper_bound1 = stock_predict1 + 1.96*sigma1
# lower_bound1 = stock_predict1 - 1.96*sigma1
# plt.fill_between(np.squeeze(x_new),lower_bound1,upper_bound1,alpha=0.5)
# plt.xlabel("Time(day)")
# plt.ylabel("Target Price(standardlized)")
# #Myfigs[0].set_title("StarBucks")
# plt.title(
#     "Kernel & Hyperparameters: %s\n"
#     % (MyGPR.kernel_)
# )
# plt.xlim((0,stock_param[1]+stock_param[2]-1))

# stock2
Myfigs[1].plot(x_train_pls1,stock_train_pls1[:,1],"r")
Myfigs[1].plot(x_new,stock_predict2,"b")
Myfigs[1].plot(x_predict_real,stock2_std[stock_param[1]:stock_param[1]+stock_param[2],:],"k")
upper_bound2 = stock_predict2 + 1.96*sigma2
lower_bound2 = stock_predict2 - 1.96*sigma2
Myfigs[1].fill_between(np.squeeze(x_new),lower_bound2,upper_bound2,alpha=0.5)
Myfigs[1].set_xlabel("Time(day)")
Myfigs[1].set_ylabel("Target Price(standardlized)")
Myfigs[1].set_title("Adobe")
Myfigs[1].set_xlim((0,stock_param[1]+stock_param[2]-1))
Myfigs[1].grid(True)

# stock3
Myfigs[2].plot(x_train_pls1,stock_train_pls1[:,2],"r")
Myfigs[2].plot(x_new,stock_predict3,"b")
Myfigs[2].plot(x_predict_real,stock3_std[stock_param[1]:stock_param[1]+stock_param[2],:],"k")
upper_bound3 = stock_predict3 + 1.96*sigma3
lower_bound3 = stock_predict3 - 1.96*sigma3
Myfigs[2].fill_between(np.squeeze(x_new),lower_bound3,upper_bound3,alpha=0.5)
Myfigs[2].set_xlabel("Time(day)")
Myfigs[2].set_ylabel("Target Price(standardlized)")
Myfigs[2].set_title("HP")
Myfigs[2].set_xlim((0,stock_param[1]+stock_param[2]-1))
Myfigs[2].grid(True)

plt.show()
