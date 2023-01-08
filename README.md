# FeatVote
An optimiser of feature sets in terms of accuracy, consistency, and generalizability 
This program optimises feature sets according to some criteria specified by the user
It utilises the experience of each feature being comined with another in reducing the time of search 
 

=====================
How to run FeatVote 
=====================
Provided you have pandas, numpy and sklearn installed on your local computer,  
and the input files are located in the same directory as the script FeatVote.py, 
follow one of the following ways 


1- From Terminal (or cmd) 
> python3 FeatVote.py train_file_name test_file_name

2- By directly running the program (e.g. F5)
The user will be propmpted to enter the name of the train file and test file 
> Enter the name of your train (required) and test (optional) csv files: 
The test file is optional 
If only file is provided it will be considered as the train file 
If no files have been provided the program will stop wit the message: No input CSV files have been specified 

The default machine learning method used in FeatVote.py is SVC 
if you need to change it you can jusy replace the first two lines of the script with your preferred model 
BUT you should keep the variable name of your model as:  model = oyur_model_definition 

