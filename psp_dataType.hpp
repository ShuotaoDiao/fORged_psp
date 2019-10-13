//
//  psp_dataType.hpp
//  fORged_psp
//
//  Created by Shuotao Diao on 10/11/19.
//  Copyright © 2019 Shuotao Diao. All rights reserved.
//

#ifndef psp_dataType_hpp
#define psp_dataType_hpp

#include <stdio.h>
#include <vector>

// ****************************************************
// Target: psp_optimization.hpp
// data structures for the dataPoint
struct dataPoint { // definition of dataPoint
    std::vector<double> predictor;
    std::vector<double> response;
    double weight;
    // default constructor
    //dataPoint();
    // copy constructor
    //dataPoint(const dataPoint& targetPoint);
    // assignment
    //dataPoint operator=(const dataPoint& targetPoint);
};

struct testResult {
    double mean = 0;
    double variance = 0;
    const double alpha = 95;
    const double Zalpha = 1.96;
    double CI_lower;
    double CI_upper;
    int num_dataPoint;
};
#endif /* psp_dataType_hpp */
