import psp_paramertric_model
import numpy as np
# data file
csvFile = "/Users/sonny/Documents/INFORMS_Project/Ten-Year-Demand.csv"
exportTXTFile = "/Users/sonny/Documents/INFORMS_Project/numerical_experiment/parametric_model/parametric_DB_p3.txt"
# time series after detrending
Z_hat, linear_model = psp_paramertric_model.data_prep(csvFile)
# print number of data points
print("Number of data points: {}".format(Z_hat))
# coefficient estimation
p = 3
coefficient, se_coefficient = psp_paramertric_model.AR_model_estimation(Z_hat, p)
# scenario generator
random_seed = 52721
N0 = 1000 # initial sample size
Delta = 50 # increment on the sample size
num_dataset = 100 # number of datasets
DB_predictor = []
DB_response = []
for index_dataset in range(num_dataset):
    N = N0 + Delta * index_dataset
    dataset_predictor = [Z_hat[-p:] for _ in range(N)]
    DB_predictor.append(dataset_predictor)
    dataset_response = list(psp_paramertric_model.scenrio_generator(Z_hat, p, linear_model, coefficient, se_coefficient,
                                                                     N, random_seed))
    DB_response.append(list(dataset_response))
# merge DB_predictor and DB_response to a database
DB = psp_paramertric_model.DB_generator(DB_predictor,DB_response)
# export DB to text file (xml-type version)
psp_paramertric_model.DB_writer(DB,exportTXTFile)

