-----------------------
Sardesai, Chaiatanya
2019-05-10
-----------------------

----------------------
Programming Language : 
----------------------
	python 3.6
	Note: Need collections, numpy, pandas, matplotlib, sklearn
	
-------------------
Package Structure :
-------------------
	This project contains 
	Code related files: creditcard.py
	Documentation: project_report.pdf
	Dataset: creditcard.csv
	Note: keep dataset at the same location as of source files *.py

------------------------
Running the Application:
------------------------ 
	1. Copy dataset file at appropriate location given above.
	2. *.py can be run as follows:
		python <python file_name>.py <dataset.csv> <number of trees> <train test split percentage>
		e.g. python creditcard.py creditcard.csv 3 70
		In case command line arguments are not given it will use creditcard.csv, 5 trees, 80% by default
	
	Note: Do not change the order of the input arguments.
	
------------
Other Notes:
------------
	Code is generic, hence more train data and test data can be added
	Results, implementation is in project_report.pdf. 
	When code executed, it run for 10 times to record 10 different results.
