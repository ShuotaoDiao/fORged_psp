import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima_model import ARIMA
#################################################
'''
Predictive Stochastic Programming for Parametric Model 
Shuotao Diao sdiao@usc.edu
Oct, 2019
with the specilization in inventory mangement 
Data is provided by Amazon Web Services
'''
#################################################
# data preparation
def data_prep(csvfile, detrend = True):
    '''

    :param csvfile:
    :param detrend: defualt is true
    :return: Z_hat N by 1 vector after detrend (if necessary)
    :return: linear_model (it is None if there is no detrending)
    '''
    linear_model = None
    # raw dataset
    hardware_device = pd.read_csv(csvfile)
    # rename the column
    hardware_device.columns = ['Year', 'Month', 'Demand']
    # number of data points
    num_dataPoint = hardware_device.shape[0]
    #
    Z = hardware_device.Demand
    Z_hat = np.array(Z)
    # if detrend is true
    if detrend:
        T = np.array([_ for _ in range(num_dataPoint)]).reshape((num_dataPoint, 1))
        linear_model = LinearRegression()
        linear_model.fit(T, Z)
        Z_trend = linear_model.predict(T)
        Z_hat = Z_hat - Z_trend
    return Z_hat, linear_model

# AR(p) model estimation
def AR_model_estimation(Z_hat, p):
    '''
    :param ts:
    :param p:
    :return: coefficients in a numpy array [c, AR1, ...,ARp]
    std_coeffcients [std_c, std_AR1, ..., std_ARp]
    '''
    AR_model = ARIMA(Z_hat,order=(p,0,0))
    AR_model_fit = AR_model.fit()
    print(AR_model_fit.params)
    coefficients = np.array([0 for _ in range(p+1)])
    se_coefficients = np.array([0 for _ in range(p+1)])
    # coefficients of the AR(p) model
    for index in range(p+1):
        coefficients[index] = AR_model_fit.params[index]
    # standard deviations of the coeffcients of the AR(p) model
    for index in range(p+1):
        se_coefficients[index] = AR_model_fit.bse[index]
    return coefficients, se_coefficients

# Scenario Generation
def scenrio_generator(Z_hat, p, linearmodel, coefficients, se_coefficients, sample_size, random_seed = None):
    '''

    :param Z_hat:
    :param p:
    :param linearmodel:
    :param coefficients:
    :param se_coefficients:
    :param sample_size:
    :param random_seed:
    :return: Z_sample  numpy array with size equal to sample_size
    '''
    # if random_seed is provided, then set the random seed
    if random_seed:
        np.random.seed(random_seed)
    coefficients_const = coefficients[0]
    se_coefficients_const = se_coefficients[0]
    # trend prediction
    T = len(Z_hat)# period for prediction
    Z_trend = linearmodel.predict(np.array([T]).reshape((1,-1)))
    # point estimator
    Z_pred = Z_hat[-p:] @ coefficients[-p:] + coefficients[0] + Z_trend
    # generate noise
    noise = np.random.normal(coefficients_const, se_coefficients_const, sample_size)
    # generate samples
    Z_sample = noise + Z_pred
    return Z_sample


# create the database suitable to predictive stochastic programming model
def DB_generator(DB_predictor, DB_response):
    '''

    :param DB_predictor: lists of lists of dictionary
    :param DB_response: lists of lists of dictionary
    :return: DB database for storing data points
    '''
    DB = []
    num_datasets = len(DB_response)
    for index_dataset in range(num_datasets):
        num_dataPoints = len(DB_response[index_dataset])
        dataset = []
        for index_dataPoint in range(num_dataPoints):
            dataPoint = {"Predictor":DB_predictor[index_dataset][index_dataPoint],
                         "Response":DB_response[index_dataset][index_dataPoint],
                         "Weight":1.0 / num_dataPoints}
            '''
            # store attributes of the predictor
            for predictor_attribute in DB_predictor[index_dataset][index_dataPoint]:
                dataPoint["Predictor"].append(predictor_attribute)
            # store the attributes of the responses
            for response_attribute in DB_response[index_dataset][index_dataPoint]:
                dataPoint["Response"].append(response_attribute)
            dataPoint["Weight"] = 1.0 / num_dataPoints # weight for
            '''
            dataset.append(dataPoint)
        DB.append(dataset)
    return DB


# output scenarios to text file
# write database
def DB_writer(database,output_path):
    '''

    :param database: lists of lists of dictionary
    :param output_path:
    :return:
    '''
    # initialization
    nameBeginDatabase = "<database>\n"
    nameEndDatabase = "</database>\n"
    nameBeginDataset = "<dataset>\n"
    nameEndDataset = "</dataset>\n"
    # number of datasets
    num_dataset = len(database)
    # write
    writeFile = open(output_path,'w')
    writeFile.write(nameBeginDatabase)
    for dataset_index in range(num_dataset):
        # begin dataset
        writeFile.write(nameBeginDataset)
        # number of data points in current dataset
        num_dataPoint = len(database[dataset_index])
        for dataPoint_index in range(num_dataPoint):
            # predictor
            writeFile.write("Predictor:")
            # size of predictor
            if isinstance(database[dataset_index][dataPoint_index]["Predictor"], list):
                predictor_size = len(database[dataset_index][dataPoint_index]["Predictor"])
                for predictor_index in range(predictor_size - 1):
                    writeFile.write("{},".format(database[dataset_index][dataPoint_index]["Predictor"][predictor_index]))
                writeFile.write("{};".format(database[dataset_index][dataPoint_index]["Predictor"][predictor_size - 1]))
            else:
                writeFile.write("{};".format(database[dataset_index][dataPoint_index]["Predictor"]))
            # response
            writeFile.write("Response:")
            # size of response
            if isinstance(database[dataset_index][dataPoint_index]["Response"], list):
                response_size = len(database[dataset_index][dataPoint_index]["Response"])
                for response_index in range(response_size - 1):
                    writeFile.write("{},".format(database[dataset_index][dataPoint_index]["Response"][response_index]))
                writeFile.write("{};".format(database[dataset_index][dataPoint_index]["Response"][response_size - 1]))
            else:
                writeFile.write("{};".format(database[dataset_index][dataPoint_index]["Response"]))
            # weight
            writeFile.write("Weight:{};\n".format(database[dataset_index][dataPoint_index]["Weight"]))
        # end dataset
        writeFile.write(nameEndDataset)
    writeFile.write(nameEndDatabase)
    writeFile.close()




