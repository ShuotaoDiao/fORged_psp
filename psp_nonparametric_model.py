import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
#################################################
'''
Predictive Stochastic Programming for Non-parametric Model 
Shuotao Diao sdiao@usc.edu
Oct, 2019
with the specilization in inventory mangement 
Data is provided by Amazon Web Services
'''
#################################################
# data preparation
def data_prep(csvfile):
    '''

    :param csvfile:
    :return: Z_hat N by 1 vector after detrend (if necessary)
    :return: linear_model (it is None if there is no detrending)
    '''
    # raw dataset
    hardware_device = pd.read_csv(csvfile)
    # rename the column
    hardware_device.columns = ['Year', 'Month', 'Demand']
    # number of data points
    num_dataPoint = hardware_device.shape[0]
    #
    Z = hardware_device.Demand
    Z_hat = np.array(Z)
    Z_hat = list(Z_hat)
    return Z_hat


# predictor response pair generator
def predictor_response_generator(Z_hat, p):
    '''
    :param Z_hat:
    :param p:
    :return:
    '''
    N = len(Z_hat)
    assert N - p > 0
    # dataset
    dataset = []
    for index in range(N - p):
        dataPoint = {"Predictor":[],
                     "Response":[],
                     "Weight":0}
        dataPoint["Response"].append(Z_hat[index + p])
        for index_predictor in range(1,p+1):
            dataPoint["Predictor"].append(Z_hat[index + p - index_predictor])
        dataset.append(dataPoint)
    return dataset


# data augmentation
def data_augmentation(dataset, N, sigma, random_seed = None):
    '''

    :param dataset: response must be one dimensional
    :param N:
    :param sigma:
    :param random_seed:
    :return:
    '''
    # if random seed is provided, then set the random seed
    if random_seed:
        np.random.seed(random_seed)
    # number of data points
    num_dataPoint = len(dataset)
    # number of attributes in the predictor
    num_attribute = len(dataset[0]["Predictor"])
    # generate a set of random indices
    random_index = np.random.choice(num_dataPoint, N)
    # generate noise
    noise = np.random.normal(0, sigma, (N,num_attribute + 1))
    # generate scenario
    dataset_aug = []
    for index in range(N):
        dataPoint = {"Predictor": [],
                     "Response": [],
                     "Weight": 0}
        for index_attribute in range(num_attribute):
            dataPoint["Predictor"].append(dataset[random_index[index]]["Predictor"][index_attribute] + noise[index, index_attribute])
        dataPoint["Response"].append(dataset[random_index[index]]["Response"][0] + noise[index, num_attribute])
        dataset_aug.append(dataPoint)
    return dataset_aug


# min max scale
def min_max_scaler(dataset):
    '''

    :param dataset: lists of dictionary
    :return: dataset_scaled, max_attribute, min_attribute
    '''
    # number of data points
    num_dataPoint = len(dataset)
    # number of attributes in the predictor
    num_attribute = len(dataset[0]["Predictor"])
    # initilization for max and min lists
    max_attribute = [float('-inf') for _ in range(num_attribute)]
    min_attribute = [float('inf') for _ in range(num_attribute)]
    for index in range(num_dataPoint):
        for index_attribute in range(num_attribute):
            if dataset[index]["Predictor"][index_attribute] > max_attribute[index_attribute]:
                # update
                max_attribute[index_attribute] = dataset[index]["Predictor"][index_attribute]
            if dataset[index]["Predictor"][index_attribute] < min_attribute[index_attribute]:
                # update
                min_attribute[index_attribute] = dataset[index]["Predictor"][index_attribute]
    # scaling
    #print(max_attribute)
    #print(min_attribute)
    dataset_scaled = []
    for index in range(num_dataPoint):
        #print("index: {}".format(index))
        dataPoint = {"Predictor":[],
                     "Response":[],
                     "Weight":0}
        for index_attribute in range(num_attribute):
            dataPoint["Predictor"].append((dataset[index]["Predictor"][index_attribute] - min_attribute[index_attribute])
                                          /(max_attribute[index_attribute] - min_attribute[index_attribute]))
        dataPoint["Response"] = dataset[index]["Response"]
        dataset_scaled.append(dataPoint)
    #print(dataset_scaled)
    print("End min max")
    return dataset_scaled, max_attribute, min_attribute


# k nearest neighbor method
def weightCal_kNN(k_beta, observed_predictor, database, DB_max_attribute, DB_min_attribute, output_path):
    print("kNN is used.")
    kNNDB = []
    # number of datasets
    num_dataset = len(database)
    # calculate weights
    for dataset_index in range(num_dataset):
        print("Dataset Index: {}".format(dataset_index))
        # number of data points in current dataset
        num_dataPoint = len(database[dataset_index])
        # calculate k
        k = math.floor(float(num_dataPoint) ** k_beta)
        assert k > 0
        # temporary set for storing distances and indices
        tempSet = []
        kNNSet = []
        # calculating squared distances
        for dataPoint_index in range(num_dataPoint):
            # scale the observed predictor
            observed_predictor_scaled = []
            for index_attribute in range(len(observed_predictor)):
                observed_predictor_scaled.append((observed_predictor[index_attribute] - DB_min_attribute[dataset_index][index_attribute])
                                                 / (DB_max_attribute[dataset_index][index_attribute]
                                                    - DB_min_attribute[dataset_index][index_attribute]))
            #print(database[dataset_index][dataPoint_index])
            distance2 = squaredDistance(observed_predictor_scaled, database[dataset_index][dataPoint_index]["Predictor"])
            tempSet.append((dataPoint_index, distance2))
            #print("Distance: {}".format(distance2))
        # calculating the weights
        tempSet_sorted = sorted(tempSet, key=lambda entry: entry[1])
        for tempSet_index in range(num_dataPoint):
            if tempSet_index < k:
                dataPoint_index = tempSet_sorted[tempSet_index][0]
                database[dataset_index][dataPoint_index]["Weight"] = 1 / k
                print("Distance: {}".format(tempSet_sorted[tempSet_index][1]))
                # print(database[dataset_index][dataPoint_index])
                kNNSet.append(database[dataset_index][dataPoint_index])  # add the point to the kNN set
            else:
                dataPoint_index = tempSet_sorted[tempSet_index][0]
                database[dataset_index][dataPoint_index]["Weight"] = 0
        # add kNN set to the kNNDB
        kNNDB.append(kNNSet)
    # write database
    DB_writer(kNNDB, output_path)

# kernel method
# calculating weights by using Epanechnikov kernel and only output the data point with positive weights
def weightCal_EpanechnikovKernel(observed_predictor, database, beta, C, DB_max_attribute, DB_min_attribute,output_path):
    print("Epanechnikov kernel is used.")
    trimDB = []
    # number of datasets
    num_dataset = len(database)
    # calculate weights
    for dataset_index in range(num_dataset):
        # number of data points in current dataset
        num_dataPoint = len(database[dataset_index])
        # temporary set
        trimSet = []
        # calculating bandwidth
        bandwidth = C * (num_dataPoint ** (-beta))
        # total weight
        totalWeight = 0
        for tempSet_index in range(num_dataPoint):
            # scale the observed predictor
            observed_predictor_scaled = []
            for index_attribute in range(len(observed_predictor)):
                observed_predictor_scaled.append((observed_predictor[index_attribute]  - DB_min_attribute[dataset_index][index_attribute])
                                                 / (DB_max_attribute[dataset_index][index_attribute]
                                                    - DB_min_attribute[dataset_index][index_attribute]))
            # temporay weight
            tempWeight = EpanechnikovKernel(observed_predictor_scaled, database[dataset_index][tempSet_index]["Predictor"], bandwidth)
            totalWeight += tempWeight
            database[dataset_index][tempSet_index]["Weight"] = tempWeight
        if totalWeight == 0:
            print("Warning: Total weight is 0, please choose larger bandwithd!")
        else:
            for tempSet_index in range(num_dataPoint):
                database[dataset_index][tempSet_index]["Weight"] = database[dataset_index][tempSet_index]["Weight"] / totalWeight
                if database[dataset_index][tempSet_index]["Weight"] > 0:
                    trimSet.append(database[dataset_index][tempSet_index])
        trimDB.append(trimSet)
    # write database
    DB_writer(trimDB, output_path)


# calculating weights by using quartic kernel and only output the data point with positive weights
def weightCal_quarticKernel(observed_predictor, database, beta, C, DB_max_attribute, DB_min_attribute,output_path):
    print("Quartic kernel is used.")
    trimDB = []
    # number of datasets
    num_dataset = len(database)
    # calculate weights
    for dataset_index in range(num_dataset):
        # number of data points in current dataset
        num_dataPoint = len(database[dataset_index])
        # temporary set
        trimSet = []
        # calculating bandwidth
        bandwidth = C * (num_dataPoint ** (-beta))
        # total weight
        totalWeight = 0
        for tempSet_index in range(num_dataPoint):
            # scale the observed predictor
            observed_predictor_scaled = []
            for index_attribute in range(len(observed_predictor)):
                observed_predictor_scaled.append((observed_predictor[index_attribute] - DB_min_attribute[dataset_index][index_attribute])
                                                 / (DB_max_attribute[dataset_index][index_attribute]
                                                    - DB_min_attribute[dataset_index][index_attribute]))
            # temporay weight
            #print(database[dataset_index][tempSet_index]["Predictor"])
            tempWeight = quarticKernel(observed_predictor_scaled, database[dataset_index][tempSet_index]["Predictor"], bandwidth)
            totalWeight += tempWeight
            database[dataset_index][tempSet_index]["Weight"] = tempWeight
        if totalWeight == 0:
            print("Warning: Total weight is 0, please choose larger bandwithd!")
        else:
            for tempSet_index in range(num_dataPoint):
                database[dataset_index][tempSet_index]["Weight"] = database[dataset_index][tempSet_index]["Weight"] / totalWeight
                if database[dataset_index][tempSet_index]["Weight"] > 0:
                    trimSet.append(database[dataset_index][tempSet_index])
        trimDB.append(trimSet)
    # write database
    DB_writer(trimDB, output_path)

# utility functions
# naive kernel
def naiveKernel(targetPredictor, predictor, bandwidth):
    value = 0
    # size of targetPredictor
    targetPredictor_size = len(targetPredictor)
    # size of predictor
    predictor_size = len(predictor)
    if targetPredictor_size != predictor_size:
        print("Warning: When calculating the naive kernel, dimensions are not equal")
    for index in range(predictor_size):
        value += (targetPredictor[index] - predictor[index]) * (targetPredictor[index] - predictor[index])
    if value <= bandwidth * bandwidth:
        return 1
    else:
        return 0


# Epanechnikov kernel
def EpanechnikovKernel(targetPredictor, predictor, bandwidth):
    value = 0
    # size of targetPredictor
    targetPredictor_size = len(targetPredictor)
    # size of predictor
    predictor_size = len(predictor)
    if targetPredictor_size != predictor_size:
        print("Warning: When calculating the naive kernel, dimensions are not equal")
    for index in range(predictor_size):
        value += (targetPredictor[index] - predictor[index]) * (targetPredictor[index] - predictor[index])
    value = value / (bandwidth * bandwidth)
    if value <= 1:
        return 1 - value
    else:
        return 0


# quartic kernel
def quarticKernel(targetPredictor, predictor, bandwidth):
    value = 0
    # size of targetPredictor
    targetPredictor_size = len(targetPredictor)
    # size of predictor
    predictor_size = len(predictor)
    if targetPredictor_size != predictor_size:
        print("Warning: When calculating the naive kernel, dimensions are not equal")
    for index in range(predictor_size):
        value += (targetPredictor[index] - predictor[index]) * (targetPredictor[index] - predictor[index])
    value = value / (bandwidth * bandwidth)
    if value <= 1:
        return (1 - value) * (1 - value)
    else:
        return 0

# kernel beta finder
def kernelBetaFinder(dimension):
    return 0.2 / dimension


def squaredDistance(a,b):
    # size of a
    a_size = len(a)
    # size of b
    b_size = len(b)
    if a_size != b_size:
        print("Warning: When calculating the distance between two points, dimensions are not equal!")
        return -1
    value = 0
    for index in range(a_size):
        value += (a[index] - b[index]) * (a[index] - b[index])
    return value


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


def sigma_estimation(Z_hat, order):
    poly_reg = PolynomialFeatures(degree=order)
    num_dataPoint = len(Z_hat)
    Z_hat_poly = np.array(Z_hat).reshape((num_dataPoint, 1))
    y = np.array([_ for _ in range(len(Z_hat))]).reshape((num_dataPoint, 1))
    y_poly = poly_reg.fit_transform(y)
    pol_reg = LinearRegression()
    pol_reg.fit(y_poly, Z_hat_poly)
    # plot
    #plt.plot(pol_reg.predict(y_poly), color = 'red')
    #plt.plot(Z_hat)
    #plt.show()
    # calculate residuals
    residual = pol_reg.predict(y_poly) - Z_hat_poly
    #
    #print(np.var(residual))
    sigma = np.sqrt(np.var(residual))
    return sigma
