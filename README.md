# fORged_psp
numpy, pandas and matplotlib are required. CPLEX is required for running cpp files. 

parametric_database_generator is used to generate training database based on the parametric model. The user needs to specify the path of the original csv file and the path for exporting training database. We use XML-type format to store the training database.

nonparametric_database_generator is used to generate training database based on the parametric model. The user needs to specify the path of the original csv file and the path for exporting training database.

ser needs to specify the the path of the training database in the main.cpp. If test database is not provided, user needs to comment out the declaration of the path for test database. 

The followings show how to link CPLEX to xcode 
link CPLEX 
Step 1: (Build Phases)
link the following libraries: 
(1) libcplex.a
(2) libcplexdistmip.a
(3) libilocplex.a
(4) libconcert.a 
(5) IOKit.framework
(6) CoreFoundation.framework

Step 2: (Build Settings)
header search path 
../CPLEX_Studio1262/concert/include 
../CPLEX_Studio1262/cplex/include

library search path 
../CPLEX_Studio1262/cplex/lib/x86-64_osx/static_pic
../CPLEX_Studio1262/concert/lib/x86-64_osx/static_pic

Step 3: (Build Settings, Customer Complier Flags)
Other C Flags: -DIL_STD

Step 4: (Build Settings, Language C++)
C++ Language Dialect: GNU++11
C++ Standard Library: libstdc++  (Now it need to be libc++)

