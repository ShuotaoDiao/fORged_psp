//
//  psp_optimization.hpp
//  fORged_psp
//
//  Created by Shuotao Diao on 10/11/19.
//  Copyright © 2019 Shuotao Diao. All rights reserved.
//

#ifndef psp_optimization_hpp
#define psp_optimization_hpp

#include <stdio.h>
#include <ilcplex/ilocplex.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include "psp_dataType.hpp"

// input and output
std::vector<std::vector<dataPoint>> readDB(std::string readPath); // read database from a text file
void printDB(const std::vector<std::vector<dataPoint>>& dataPointDB); // print dataPoint

// model setup
double psp_inventory_model(const std::vector<std::vector<dataPoint>>& dataPointDB, double y0); // 
std::vector<testResult> psp_inventory_model_test(const std::vector<std::vector<dataPoint>>& dataPointDB, double y0, double x_est); // estimate average cost from the test set
// test on the linking to the cplex solver
void test_linking_cplex();
#endif /* psp_optimization_hpp */
