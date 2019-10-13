//
//  psp_optimization.cpp
//  fORged_psp
//
//  Created by Shuotao Diao on 10/11/19.
//  Copyright © 2019 Shuotao Diao. All rights reserved.
//

#include "psp_optimization.hpp"

std::vector<std::vector<dataPoint>> readDB(std::string readPath){
    std::cout << "Read Database" << std::endl;
    std::vector<std::vector<dataPoint>> dataPointDB; // database
    std::vector<dataPoint> dataPointDS; // dataset
    std::string namePredictor("Predictor");
    std::string nameWeight("Weight");
    std::string nameBeginDB("<database>");
    std::string nameEndDB("</database>");
    std::string nameBeginDS("<dataset>");
    std::string nameEndDS("</dataset>");
    bool flag_ifDS = false; // flag of whether the current dataset has been reading
    const char* readPathConst = readPath.c_str(); // convert the string type path to constant
    std::ifstream readFile(readPathConst); // create a readFile object
    if (readFile.is_open()) {
        std::string line1;
        while (getline(readFile, line1)) { // get the whole line
            std::stringstream ss1(line1); // convert a string into stream
            dataPoint dataPointTemp;
            unsigned int index_position = 0; // 1 for predictor, 3 for response, 5 for weight
            if (nameBeginDB.compare(line1) != 0 && nameEndDB.compare(line1) != 0) { // main
                // contents of DB
                //std::cout << line1 << std::endl;
                //std::cout << "Main Content" << std::endl;
                if (nameBeginDS.compare(line1) == 0 && !flag_ifDS) {// if it is a new dataset
                    //std::cout << "Begin of Dataset" << std::endl;
                    flag_ifDS = true; // start reading new dataset and set the flag to be true
                }
                else if (flag_ifDS && nameEndDS.compare(line1) != 0){ // if it is the current dataset
                    //std::cout << "Data Point" << std::endl;
                    while (getline(ss1, line1, ';')) {
                        std::stringstream ss2(line1);
                        while (getline(ss2, line1, ':')) {
                            if (index_position == 1){ // read vector
                                std::stringstream ss3(line1);
                                while (getline(ss3, line1, ',')) {
                                    double value;
                                    std::stringstream ss4(line1);
                                    ss4 >> value;
                                    dataPointTemp.predictor.push_back(value);
                                }
                            }
                            else if (index_position == 3){
                                double value;
                                std::stringstream ss3(line1);
                                while (getline(ss3, line1, ',')) {
                                    std::stringstream ss4(line1);
                                    ss4 >> value;
                                    dataPointTemp.response.push_back(value);
                                }
                            }
                            else if (index_position == 5){
                                double value;
                                std::stringstream ss3(line1);
                                ss3 >> value;
                                dataPointTemp.weight = value;
                            }
                            index_position++;
                        }
                    }
                    dataPointDS.push_back(dataPointTemp);
                }
                else if (nameEndDS.compare(line1) == 0){ // if it is the end of a dataset
                    //std::cout << "End of Dataset" << std::endl;
                    flag_ifDS = false; // finishing reading current dataset and set the flag to be false
                    dataPointDB.push_back(dataPointDS); // add the dataset the database
                    dataPointTemp = {}; // clear a struct
                    dataPointDS.clear(); // clear the temporary dataset
                }
            }
        }
    }
    readFile.close(); // close the file
    return dataPointDB;
}

void printDB(const std::vector<std::vector<dataPoint>>& dataPointDB){
    long DS_number = dataPointDB.size();
    std::cout << DS_number << std::endl;
    for (int DS_index = 0; DS_index < DS_number; DS_index++) {
        long dataPoint_number = dataPointDB[DS_index].size();
        std::cout << "Dataset: " << DS_index + 1 << std::endl;
        for (int dataPoint_index = 0; dataPoint_index < dataPoint_number; dataPoint_index++) {
            std::cout << "Predictor " << dataPoint_index + 1 << ": ";
            long predictor_size = dataPointDB[DS_index][dataPoint_index].predictor.size(); // size of predictor
            for (int predictor_index = 0; predictor_index < predictor_size; ++predictor_index) {
                std::cout << dataPointDB[DS_index][dataPoint_index].predictor[predictor_index] << ", ";
            }
            long response_size = dataPointDB[DS_index][dataPoint_index].response.size(); // size of predictor
            std::cout << "Response " << dataPoint_index + 1 << ": ";
            for (int response_index = 0; response_index < response_size; response_index++) {
                std::cout << dataPointDB[DS_index][dataPoint_index].response[response_index] << ", ";
            }
            std::cout << "Weight: " << dataPointDB[DS_index][dataPoint_index].weight << std::endl;
        }
    }
}


// model setup
// inventory management model
double psp_inventory_model(const std::vector<std::vector<dataPoint>>& dataPointDB, double y0) {
    // initialize the model
    long num_dataset = dataPointDB.size(); // number of datasets
    double x_opt = 0;
    for (int iteration = 0; iteration < num_dataset; ++iteration) {
        long num_dataPoint = dataPointDB[iteration].size(); // number of data points in the current dataset
        // cplex environment
        IloEnv env;
        IloModel mod(env);
        IloNumVar x(env,0,IloInfinity,ILOFLOAT);
        // second stage
        IloNumVarArray y1(env,num_dataPoint,0,IloInfinity,ILOFLOAT); // invetory at the end of the month
        IloNumVarArray s(env,num_dataPoint,0,IloInfinity,ILOFLOAT); // backorder
        // t is the max function
        IloNumVarArray u(env,num_dataPoint,0,IloInfinity,ILOFLOAT); //
        // add the decision variables to the model
        mod.add(x);
        mod.add(y1);
        mod.add(s);
        mod.add(u);
        // expression for the objective function
        IloExpr expr_obj(env);;
        // objective function and constraints
        for (int index_dataPoint = 0; index_dataPoint < num_dataPoint; ++index_dataPoint) {
            expr_obj += dataPointDB[iteration][index_dataPoint].weight * (y1[index_dataPoint] + u[index_dataPoint] + 3.0 * s[index_dataPoint]);
            // constraints
            // balance constraint
            IloExpr expr_balance(env);
            expr_balance = y1[index_dataPoint] - x - s[index_dataPoint];
            mod.add(expr_balance == y0 - dataPointDB[iteration][index_dataPoint].response[0]);
            // constraint for the max function
            IloExpr expr_max(env);
            expr_max = u[index_dataPoint] - y1[index_dataPoint];
            mod.add(expr_max >= -90);
        }
        IloObjective obj = IloMinimize(env, expr_obj);
        mod.add(obj);
        // cplex solver setup
        IloCplex cplex(env);
        cplex.extract(mod);
        cplex.solve();
        // print optimal solution
        std::cout << "===========================================================\n";
        std::cout << "Numerical Results\n";
        std::cout << "Iteration: " << iteration << std::endl;
        std::cout << "Initial inventory: " << y0 << std::endl;
        std::cout << "Optimal solution (replenishment): " << cplex.getValue(x) << std::endl;
        x_opt = cplex.getValue(x);
        // print average cost
        std::cout << "Average cost: " << cplex.getObjValue() << std::endl;
        // print average holding cost
        double average_holding_cost = 0;
        double average_backorder_cost = 0;
        for (int index = 0; index < num_dataPoint; index++) {
            average_holding_cost += (cplex.getValue(y1[index]) + cplex.getValue(u[index])) / num_dataPoint;
            average_backorder_cost += 3.0 * cplex.getValue(s[index]) / num_dataPoint;
        }
        std::cout << "Average holding cost: " << average_holding_cost << std::endl;
        std::cout << "Average backorder cost: " << average_backorder_cost << std::endl;
        std::cout << "===========================================================\n";
        env.end();
    }
    return x_opt;
} // end psp_inventory_model

// estimate average cost from the test set
std::vector<testResult> psp_inventory_model_test(const std::vector<std::vector<dataPoint>>& dataPointDB, double y0, double x_est) {
    long num_dataPoint = dataPointDB[0].size();
    // cplex environment
    IloEnv env;
    IloModel mod(env);
    // second stage
    IloNumVarArray y1(env,num_dataPoint,0,IloInfinity,ILOFLOAT); // invetory at the end of the month
    IloNumVarArray s(env,num_dataPoint,0,IloInfinity,ILOFLOAT); // backorder
    // t is the max function
    IloNumVarArray u(env,num_dataPoint,0,IloInfinity,ILOFLOAT); //
    // add the decision variables to the model
    mod.add(y1);
    mod.add(s);
    mod.add(u);
    // expression for the objective function
    IloExpr expr_obj(env);;
    // objective function and constraints
    for (int index_dataPoint = 0; index_dataPoint < num_dataPoint; ++index_dataPoint) {
        expr_obj += dataPointDB[0][index_dataPoint].weight * (y1[index_dataPoint] + u[index_dataPoint] + 3.0 * s[index_dataPoint]);
        // constraints
        // balance constraint
        IloExpr expr_balance(env);
        expr_balance = y1[index_dataPoint] - x_est - s[index_dataPoint];
        mod.add(expr_balance == y0 - dataPointDB[0][index_dataPoint].response[0]);
        // constraint for the max function
        IloExpr expr_max(env);
        expr_max = u[index_dataPoint] - y1[index_dataPoint];
        mod.add(expr_max >= -90);
    }
    IloObjective obj = IloMinimize(env, expr_obj);
    mod.add(obj);
    // cplex solver setup
    IloCplex cplex(env);
    cplex.extract(mod);
    cplex.solve();
    // calculate test results
    testResult cost;
    cost.mean = cplex.getObjValue();
    cost.num_dataPoint = num_dataPoint;
    testResult holding_cost;
    holding_cost.num_dataPoint = num_dataPoint;
    testResult backorder_cost;
    backorder_cost.num_dataPoint = num_dataPoint;
    double cost_variance_p1 = 0;
    double holding_cost_variance_p1 = 0;
    double backorder_cost_variance_p1 = 0;
    // calculate mean and variance
    for (int index = 0; index < num_dataPoint; ++index) {
        holding_cost.mean += (cplex.getValue(y1[index]) + cplex.getValue(u[index])) / num_dataPoint;
        backorder_cost.mean += 3.0 * cplex.getValue(s[index]) / num_dataPoint;
        int _n_ = index + 1;
        double holding_cost_scenario = cplex.getValue(y1[index]) + cplex.getValue(u[index]);
        double backorder_cost_scenario = 3.0 * cplex.getValue(s[index]);
        double cost_scenario = holding_cost_scenario + backorder_cost_scenario;
        // update
        cost_variance_p1 = cost_variance_p1 * (_n_ - 1) / _n_ + cost_scenario * cost_scenario / _n_;
        holding_cost_variance_p1 = holding_cost_variance_p1 * (_n_ - 1) / _n_  + holding_cost_scenario * holding_cost_scenario / _n_;
        backorder_cost_variance_p1 = backorder_cost_variance_p1 * (_n_ - 1) / _n_  + backorder_cost_scenario * backorder_cost_scenario / _n_;
    }
    std::cout << cost_variance_p1 << std::endl;
    cost.variance = cost_variance_p1 * ((double) num_dataPoint / (num_dataPoint - 1.0)) - cost.mean * cost.mean * ((double) num_dataPoint / (num_dataPoint - 1.0));
    holding_cost.variance = holding_cost_variance_p1 * ((double) num_dataPoint / (num_dataPoint - 1.0)) - holding_cost.mean * holding_cost.mean * ((double) num_dataPoint / (num_dataPoint - 1.0));
    backorder_cost.variance = backorder_cost_variance_p1 * ((double) num_dataPoint / (num_dataPoint - 1.0)) - backorder_cost.mean * backorder_cost.mean * ((double) num_dataPoint / (num_dataPoint - 1.0));
    // confidence interval
    double cost_halfMargin = cost.Zalpha * sqrt(cost.variance / num_dataPoint);
    double holding_cost_halfMargin = holding_cost.Zalpha * sqrt(holding_cost.variance / num_dataPoint);
    double backorder_cost_halfMargin = backorder_cost.Zalpha * sqrt(backorder_cost.variance / num_dataPoint);
    cost.CI_lower = cost.mean - cost_halfMargin;
    cost.CI_upper = cost.mean + cost_halfMargin;
    holding_cost.CI_lower = holding_cost.mean - holding_cost_halfMargin;
    holding_cost.CI_upper = holding_cost.mean + holding_cost_halfMargin;
    backorder_cost.CI_lower = backorder_cost.mean - backorder_cost_halfMargin;
    backorder_cost.CI_upper = backorder_cost.mean + backorder_cost_halfMargin;
    std::vector<testResult> res;
    res.push_back(cost);
    res.push_back(holding_cost);
    res.push_back(backorder_cost);
    env.end();
    return res;
}

// test on the linking to the cplex solver
void test_linking_cplex() {
    IloEnv env;
    IloModel mod(env);
    IloNumVar x(env,-1,1,ILOFLOAT);
    mod.add(x);
    IloExpr expr(env);
    expr = x * x - 0.5 * x;
    IloObjective obj = IloMinimize(env, expr);
    mod.add(obj);
    IloCplex cplex(env);
    cplex.extract(mod);
    cplex.solve();
    std::cout << "Optimal cost is " << cplex.getObjValue() << std::endl;
    std::cout << "Optimal solution is " << cplex.getValue(x) << std::endl;
    env.end();
}
