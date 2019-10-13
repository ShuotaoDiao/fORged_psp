//
//  main.cpp
//  fORged_psp
//
//  Created by Shuotao Diao on 10/11/19.
//  Copyright © 2019 Shuotao Diao. All rights reserved.
//

#include <iostream>
#include <ilcplex/ilocplex.h>
#include <vector>
#include "psp_optimization.hpp"

int main(int argc, const char * argv[]) {
    //=================================================================================
    // please specify the paths of the training database and test database
    //std::string read_DB_file = "/Users/sonny/Documents/INFORMS_Project/numerical_experiment/parametric_model/parametric_DB_p2.txt";
    std::string read_DB_file = "/Users/sonny/Documents/INFORMS_Project/numerical_experiment/nonparametric_model/kNN/nonparametric_DB_p2.txt";
    std::string read_DB_test_file = "/Users/sonny/Documents/INFORMS_Project/numerical_experiment/nonparametric_model/kNN/nonparametric_DB_test.txt";
    double y0 = 20;
    //=================================================================================
    std::vector<std::vector<dataPoint>> DB = readDB(read_DB_file);
    std::vector<std::vector<dataPoint>> DB_test = readDB(read_DB_test_file);
    //printDB(DB);
    //
    double x_est = psp_inventory_model(DB, y0);
    //double x_est = 85;
    std::vector<testResult> res = psp_inventory_model_test(DB_test, y0, x_est);
    // print test results
    // cost
    std::cout << "Number of Data Points: " << res[0].num_dataPoint << std::endl;
    std::cout << ">>>>Cost<<<<" << std::endl;
    std::cout << "Average cost: " << res[0].mean << std::endl;
    std::cout << "95% confidence interval of the expected cost: [" << res[0].CI_lower << "," << res[0].CI_upper << "]" << std::endl;
    std::cout << ">>>>Holding Cost<<<<" << std::endl;
    std::cout << "Average holidng cost: " << res[1].mean << std::endl;
    std::cout << "95% confidence interval of the expected holidng cost: [" << res[1].CI_lower << "," << res[1].CI_upper << "]" << std::endl;
    std::cout << ">>>>Backorder Cost<<<<" << std::endl;
    std::cout << "Average cost: " << res[2].mean << std::endl;
    std::cout << "95% confidence interval of the expected backorder cost: [" << res[2].CI_lower << "," << res[2].CI_upper << "]" << std::endl;
    
    return 0;
}
