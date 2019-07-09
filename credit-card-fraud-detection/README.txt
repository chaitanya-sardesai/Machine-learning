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
Dataset: creditcard.csv (can be found at kaggle.com)
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

----------------
Major Functions:
----------------
Function name: main()
Description: Main function used to call all other required subroutines to execute various tasks. Generates Confusion matrix after test data classification and computes Recall and accuracy.

Function name: normalize_feature(feature_name, range_of_rescaling)
Description: Rescale normalization technique is used, returns feature column rescaled in place.

Function Name: split_train_test_data(data, percentage)
Description: Based on percentage, dataset is divided into train data and test data with the given percenatge of data as train data. 
returns train_data, test_data

Function Name: balanced_data(data)
Description: calculate number of samples for both classes. Takes all samples of class having lower number of samples than other. Choose random samples from pool of other class to match the number of samples to 1st class. i.e. equalizes number of samples of both classes. returns equalized dataset.
Note: No need to call this function if dataset is not extremly unbalanced

Function Name: dtl(tree_node, data, prob_distribution, features_lst)
Description: Train and build decision tree based on given train data and features list. At each node probability distribution of class is stored. returns root node.

Function Name: get_best_attribute(data, feature_list)
Description: Pick random feature from feature list, and calculates best possible threshold for that feature based on information gain (entropy). It uses 50 threshold between min_value and max_value of the feature. returns feature and threshold of maximum information gain.

Function Name: calc_prob_distribution(data)
Description: Calculates probability distribution based on class examples present in current data. return probability distribution in array.

Function Name: calc_information_gain(data, feature_data, threshold)
Description: Creates left subtree and right subtree based on the threshold. Calculates information gain using root node number of examples, and number of examples from left subtree and right subtree. returns information gain for that feature, for given threshold.

Function Name: calc_entropy(node)
Description: Given node, calculate its entropy using probability of classes.

Function Name: check_same_class(data)
Description: Check if all the examples in data have same class. return True if all examples have same class, otherwise False

Function Name: recursive_classify(tree_node, row_data)
Description: Traverse decision tree to leaf nodes based on the features and thresholds at every node of the tree given the current example data(row data). Retrives the probability distribution from the leaf node and returns it.

Function Name: classify_using_dtl(root, test_data)
Description: Classify examples from test data to classes using trained decision tree. Internally calls recursive_classify(tree_node row_data). return list containing class of every examples of the test data.

------------
Other Notes:
------------
Code is generic, hence more train data and test data can be added.
	
