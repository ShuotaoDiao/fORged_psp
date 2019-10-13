# fORged_psp
numpy, pandas and matplotlib are required
CPLEX is required for running cpp files. 

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

