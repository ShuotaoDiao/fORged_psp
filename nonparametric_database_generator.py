import psp_nonparametric_model
import numpy as np
import os
#################################################################################################################
# path of data file
csvFile = "/Users/sonny/Documents/INFORMS_Project/Ten-Year-Demand.csv"
#method = "kNN"
method = "quartic"
# output path
exportTXTFile = "/Users/sonny/Documents/INFORMS_Project/numerical_experiment/nonparametric_model/"+ \
                method +"/parametric_DB_p2_new.txt"
# set up initial parameters
random_seed = 721285
p = 2 # number of periods look back
N0 = 100
Delta = 5
num_dataset = 2 # number of datasets
#--------------------
# parameter for kNN
k_beta = 0.6
#--------------------
#--------------------
# paramters for kernel method
beta = 0.2 / p
C = 1.5
#--------------------
#################################################################################################################
# create a folder if it does not exist
directory = os.path.dirname(exportTXTFile)
if not os.path.exists(directory):
    os.makedirs(directory)

# import dataset
Z_hat = psp_nonparametric_model.data_prep(csvFile)
observed_predictor = Z_hat[-p:]
num_attributes = len(observed_predictor)
print(np.var(Z_hat))
sigma = psp_nonparametric_model.sigma_estimation(Z_hat,order = 5)
print("Sigma: ", sigma)

# predictor response dataset
dataset = psp_nonparametric_model.predictor_response_generator(Z_hat,p)

# generate database
database_scaled = []
DB_max_attribute = []
DB_min_attribute = []
# generate database
for index_dataset in range(num_dataset):
    N = N0 + Delta * index_dataset
    # dataset augmentation
    dataset_aug = psp_nonparametric_model.data_augmentation(dataset, N, sigma, random_seed)
    print("Dataset augment")
    print(dataset_aug)
    # min max scale for the predictor
    dataset_scaled, max_attribute, min_attribute = psp_nonparametric_model.min_max_scaler(dataset_aug)
    print("Dataset scaled")
    print(dataset_scaled)
    # store the dataset
    database_scaled.append(dataset_scaled)
    DB_max_attribute.append(max_attribute)
    DB_min_attribute.append(min_attribute)

# weights are calculated by kNN
if method == "kNN":
    psp_nonparametric_model.weightCal_kNN(k_beta, observed_predictor, database_scaled,
                                          DB_max_attribute, DB_min_attribute, exportTXTFile)
if method == "quartic":
    psp_nonparametric_model.weightCal_quarticKernel(observed_predictor,database_scaled, beta,
                                                    C, DB_max_attribute, DB_min_attribute,exportTXTFile)
