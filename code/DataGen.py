# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
import importlib

"""
this file contains object to create data sets for regression

synthetic datasets: 
drunk_bow_tie - as in paper, with gaussian noise
drunk_bow_tie_exp - as in paper with exp noise
x_cubed_gap - as in paper to show model uncertainty

real datasets:
~boston - standard boston housing dataset

"""


class DataGenerator:
	def __init__(self, type_in, n_feat=1):
		# select type of data to produce
		# not really using no. feat anymore

		self.n_feat = n_feat
		self.type_in = type_in
		
		return
		
	
		
		
		
		
	
	def CreateData(self, n_samples, seed_in=5, 
		train_prop=0.9, bound_limit=6., n_std_devs=1.96,**kwargs):
		

			
		np.random.seed(seed_in)
		scale_c=1.0 # default
		shift_c=1.0
		
		# for ideal boundary
		X_ideal = np.linspace(start=-bound_limit,stop=bound_limit, num=50000)
		y_ideal_U = np.ones_like(X_ideal)+1. # default
		y_ideal_L = np.ones_like(X_ideal)-1.
		y_ideal_mean = np.ones_like(X_ideal)+0.5

		
		
# 		nonlocal X_train, y_train, X_val, y_val ## ADDED BY PARUL
		
# 		if self.type_in=="drunk_bow_tie":
# 			"""
# 			similar to bow tie but less linear
# 			"""	
		
# 			X = np.random.uniform(low=-2.,high=2.,size=(n_samples,1))
# 			y = 1.5*np.sin(np.pi*X[:,0]) + np.random.normal(loc=0.,scale=1.*np.power(X[:,0],2))
# 			y = y.reshape([-1,1])/5.
# 			X_train = X
# 			y_train = y	

# 			X = np.random.uniform(low=-2.,high=2.,size=(int(10*n_samples),1))
# 			y = 1.5*np.sin(np.pi*X[:,0]) + np.random.normal(loc=0.,scale=1.*np.power(X[:,0],2))
# 			y = y.reshape([-1,1])/5.		
# 			X_val = X
# 			y_val = y

# 			y_ideal_U = 1.5*np.sin(np.pi*X_ideal) + n_std_devs*np.power(X_ideal,2)
# 			y_ideal_U = y_ideal_U/5.
# 			y_ideal_L = 1.5*np.sin(np.pi*X_ideal) - n_std_devs*np.power(X_ideal,2)
# 			y_ideal_L = y_ideal_L/5.
# 			y_ideal_mean = 1.5*np.sin(np.pi*X_ideal)
# 			y_ideal_mean = y_ideal_mean/5.	

# 			# overwrite for convenience!
# 			X_val = X_train
# 			y_val = y_train

# 		elif self.type_in=="drunk_bow_tie_exp":
# 			"""
# 			similar to bow tie but less linear, now with non-gaussian noise
# 			"""	

# 			X = np.random.uniform(low=-2.,high=2.,size=(n_samples,1))
# 			y = 1.5*np.sin(np.pi*X[:,0]) + np.random.exponential(scale=1.*np.power(X[:,0],2))
# 			y = y.reshape([-1,1])/5.
# 			X_train = X
# 			y_train = y	

# 			X = np.random.uniform(low=-2.,high=2.,size=(int(10*n_samples),1))
# 			y = 1.5*np.sin(np.pi*X[:,0]) + np.random.exponential(scale=1.*np.power(X[:,0],2))
# 			y = y.reshape([-1,1])/5.		
# 			X_val = X
# 			y_val = y

# 			# for exponential quantile = ln(1/quantile) /lambda
# 			# note that np inputs beta = 1/lambda
# 			y_ideal_U = 1.5*np.sin(np.pi*X_ideal) + np.log(1/(1-0.95))*np.power(X_ideal,2)
# 			y_ideal_U = y_ideal_U/5.
# 			y_ideal_L = 1.5*np.sin(np.pi*X_ideal)
# 			y_ideal_L = y_ideal_L/5.
# 			y_ideal_mean = 1.5*np.sin(np.pi*X_ideal)
# 			y_ideal_mean = y_ideal_mean/5.	

# 			X_val = X_train
# 			y_val = y_train

# 		elif self.type_in=="periodic_1":
# 			"""
# 			creates a bow tie shape with changing variance
# 			"""
# 			X = np.random.uniform(low=-5.,high=5.,size=(n_samples,self.n_feat))
# 			y = 2.1*np.cos(0.2*X[:,0]) + 0.7*np.cos(20.1*X[:,0]) + 0.2*np.cos(10.4*X[:,0]) + np.random.normal(loc=0.,scale=0.1*np.ones_like(X[:,0]))
# 			y = y.reshape([-1,1])/1.
# 			X_train = X
# 			y_train = y	
# 			X_val = X_train
# 			y_val = y_train
# 			# y_ideal_U = X_ideal/5. + n_std_devs * np.abs(X_ideal)/5.
# 			# y_ideal_L = X_ideal/5. - n_std_devs * np.abs(X_ideal)/5.

# 		elif self.type_in=="x_cubed_gap":
# 			"""
# 			toy data problem from Probabilistic Backprop (Lobato) & 
# 			deep ensembles (Blundell)
# 			but added gap here

# 			"""
# 			scale_c = 50.
# 			half_samp = int(round(n_samples/2))
# 			X_1 = np.random.uniform(low=-4.,high=-1.,size=(half_samp,1))
# 			X_2 = np.random.uniform(low=1.,high=4.,size=(n_samples - half_samp,1))
# 			X = np.concatenate((X_1, X_2))
# 			y = X[:,0]**3 + np.random.normal(loc=0.,scale=3., size=X[:,0].shape[0])
# 			y = y.reshape([-1,1])/scale_c
# 			X_train = X
# 			y_train = y			
# 			X_val = X_train
# 			y_val = y_train

# 			y_ideal_U = X_ideal**3 + n_std_devs*3.
# 			y_ideal_U = y_ideal_U/scale_c
# 			y_ideal_L = X_ideal**3 - n_std_devs*3.
# 			y_ideal_L = y_ideal_L/scale_c
# 			y_ideal_mean = X_ideal**3
# 			y_ideal_mean = y_ideal_mean/scale_c

		# use single char '~' at start to identify real data sets
# 		elif self.type_in[:1] == '~':
# 		if self.type_in[:1] == '~':
# 			if self.type_in=="~boston":
# 				path = 'boston_housing_data.csv'
# 				data = np.loadtxt(path,skiprows=0)
# 			elif self.type_in=="~concrete":
# 				path = 'Concrete_Data.csv'
# 				data = np.loadtxt(path, delimiter=',',skiprows=1)
		if self.type_in=="~wind":
			path = '/content/Deep_Learning_Prediction_Intervals/code/canada_CSV.csv'
			data = np.loadtxt(path,delimiter=',',skiprows=1,usecols = (1,2)) ## CHECK WHTHER TO HAVE LOADTXT OR ANYTHING ELSE PARUL
			
			
			# work out normalisation constants (need when unnormalising later)
		scale_c = np.std(data[:,-1])
		shift_c = np.mean(data[:,-1])

			# normalise data for ALL COLUMNS
			for i in range(0,data.shape[1]): ## i varies from 0 to number of columns ,means it reads one by one the columns
				# avoid zero variance features (exist one or two)

				sdev_norm = np.std(data[:,i])
				sdev_norm = 0.001 if sdev_norm == 0 else sdev_norm
				data[:,i] = (data[:,i] - np.mean(data[:,i]) )/sdev_norm
			
# 			global X_train, y_train, X_val, y_val ## ADDED BY PARUL
			# split into train/test
			perm = np.random.permutation(data.shape[0]) ## DO THE DATA PERMUTATION OF ALL THE ROWS (shuffle)
			train_size = int(round(train_prop*data.shape[0]))
			train = data[perm[:train_size],:]
			test = data[perm[train_size:],:]
			
			
			y_train = train[:,-1].reshape(-1,1) ## LAST COLUMN IS CONSIDERED AS THE TARGET AND RESHAPED IN BETWEEN -1,1
			X_train = train[:,:-1] ## INPUTS ARE ALL EXCEPT LAST COLUMN
			y_val = test[:,-1].reshape(-1,1)
			X_val = test[:,:-1]

			# save important stuff
			self.X_train = X_train
			self.y_train = y_train
			self.X_val = X_val
			self.y_val = y_val
			self.X_ideal = X_ideal
			self.y_ideal_U = y_ideal_U
			self.y_ideal_L = y_ideal_L
			self.y_ideal_mean = y_ideal_mean
			self.scale_c = scale_c
			self.shift_c = shift_c

		return X_train, y_train, X_val, y_val


	def ViewData(self, n_rows=5, hist=True, plot=True, print_=True):#plot and hist changed by PARUL
		"""
		print first few rows of data
		option to view histogram of x and y
		option to view scatter plot of x vs y
		"""
		if print_:
			print("\nX_train\n",self.X_train[:n_rows], 
				"\ny_train\n", self.y_train[:n_rows], 
				"\nX_val\n", self.X_val[:n_rows], 
				"\ny_val\n", self.y_val[:n_rows])

		if hist:
			fig, ax = plt.subplots(1, 2)
			ax[0].hist(self.X_train)
			ax[1].hist(self.y_train)
			ax[0].set_title("X_train")
			ax[1].set_title("y_train")
			fig.show()

		if plot:
			n_feat = self.X_train.shape[1]
			fig, ax = plt.subplots(n_feat, 1) # create an extra
			if n_feat == 1:	ax = [ax] # make into list
			for i in range(0,n_feat):
				ax[i].scatter(self.X_train[:,i],self.y_train,
					alpha=0.5,s=2.0)
				ax[i].set_xlabel('x_'+str(i))
				ax[i].set_ylabel('y')

			fig.show()

		return

		
