import psp_nonparametric_model
import numpy as np
import os
import math
#################################################################################################################
# data file
# Please modify the parameters or paths if necessary in this block
csvFile = "/Users/sonny/Documents/INFORMS_Project/Ten-Year-Demand.csv"
method = "kNN" # "Epanechnikov" "Quartic"
p = 2 # number of periods look back
exportDB = "/Users/sonny/Documents/INFORMS_Project/numerical_experiment/nonparametric_model/" + \
                method + "/nonparametric_DB_p{}.txt".format(p)
exportDB_test = "/Users/sonny/Documents/INFORMS_Project/numerical_experiment/nonparametric_model/" + \
                method + "/nonparametric_DB_test.txt"
if_output_DBtest = True
# paramters of scenario generator
random_seed = 721285
N0 = 1000
Delta = 50
num_dataset = 100 # number of datasets
#--------------------
# parameter for kNN
k_beta = 0.6
#--------------------
#--------------------
# paramters for kernel method
beta = 0.2 / p
C = 1.5
#--------------------
index_test = 100 # index for testing must be smaller than 120 - p
test_set_size = 10000
#################################################################################################################
# create a folder if it does not exist
directory = os.path.dirname(exportDB)
if not os.path.exists(directory):
    os.makedirs(directory)

# import dataset
Z_hat = psp_nonparametric_model.data_prep(csvFile)
print(np.var(Z_hat))
sigma = psp_nonparametric_model.sigma_estimation(Z_hat,order = 5)

# predictor response dataset
dataset = psp_nonparametric_model.predictor_response_generator(Z_hat,p)
# observe predictor based on the test_index
observed_predictor = dataset[index_test]["Predictor"]
# create training set and test set
num_dataPoints = len(dataset)
print("Number of data points: ", num_dataPoints)
num_dataPoints_train = index_test
dataset_train = []
dataset_test = []
for index in range(num_dataPoints_train):
    dataset_train.append(dataset[index])
np.random.seed(random_seed)
noise_test = np.random.normal(0,sigma,test_set_size)
for index in range(test_set_size):
    dataPoint = {"Predictor": dataset[index_test]["Predictor"],
                 "Response": float(dataset[index_test]["Response"] + noise_test[index]),
                 "Weight": 1.0 / test_set_size}
    dataset_test.append(dataPoint)
DB_test = [dataset_test]
# output test database
if if_output_DBtest:
    psp_nonparametric_model.DB_writer(DB_test, exportDB_test)

# data augmentation on the training set
# generate database
database_scaled = []
DB_max_attribute = []
DB_min_attribute = []
# generate database
for index_dataset in range(num_dataset):
    N = N0 + Delta * index_dataset
    # dataset augmentation
    dataset_aug = psp_nonparametric_model.data_augmentation(dataset_train, N, sigma, random_seed)
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
                                          DB_max_attribute, DB_min_attribute, exportDB)





