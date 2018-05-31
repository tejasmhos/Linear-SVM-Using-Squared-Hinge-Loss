# Linear SVM Using Squared Hinge Loss
This is an implementation of a Linear SVM that uses a squared hinge loss. This algorithm was coded using Python. This is my submission for the polished code release for DATA 558 - Statistical Machine Learning. The code was developed by Tejas Hosangadi.

## The Optimization Problem

The Linear SVM that Uses Squared Hinge Loss writes out as shown below:

![equation](https://raw.githubusercontent.com/tejasmhos/Linear-SVM-Using-Squared-Hinge-Loss/master/equation.jpeg)

The above equation is differentiable and convex, hence we can apply gradient descent. This implementation of the SVM uses the fast gradient algorithm, which improves the speed and accuracy of the descent. As is expected, the SVM is a binary classifier, and can be used to perform classification of data that exists in two classes. If more than two classes exist, multi class classification strategies like one versus one (1v1) and one versus rest (1vr) can be used.

The algorithm uses backtracking to determine the best step size. In addition, the algorithm stops when the maximum number of iterations are reached.



## Directory Structure

The structure of the directory is shown below:

```
|- Linear-SVM-Using-Squared-Hinge-Loss\
	|- linearsvm\
		|- linearsvm.py
		|- real_data_demo.py
		|- simulated_data_demo.py
		|- sklearn_compare.py
	|- LICENSE
	|- README.md
	|- equation.jpeg
```

1. **linearsvm.py:** This contains all the coded methods required by the project. If you want to understand how I coded my methods, you can have a look at the file. It is not explicitly executable, but contains methods used by the other modules.
2. **real_data_demo.py:** This module contains the code that demonstrates the SVM for real data, in this case, spam data from The Elements of Statistical Learning.
3. **simulated_data_demo.py:** This module contains the code that demonstrates the SVM for synthetic/simulated data.
4. **sklearn_compare.py:** This module contains code that checks the performance of our SVM compared to the one that is implemented by sklearn.



## Dependencies 

This code uses a number of different dependencies. First and foremost, Python 3 is required. Most Unix based systems (including Mac), include Python installed. It's always ideal to check to make sure the correct version is installed.

The Python based packages that are required are listed below

1. Numpy
2. Scipy
3. Scikit-learn
4. Pandas

These can be installed using the pip command or the conda command if an Anaconda based environment is being used.



## Data

Two types of data are used in the demo files, one is a simulated dataset, and the other is the spam dataset from The Elements of Statistical Learning. All data is downloaded on execution, and you are not required to manually download or install any data.



## Instructions

First and foremost, clone or download the repository to your local system. Make sure all the dependencies are installed, and work as they are supposed to. Finally, execute the demo and sklearn comparison modules as shown below from your terminal:

```
$>python name_of_the_module.py
```

On execution, you will get some textual output in the terminal window, and a series of plots will be generated. Plots are generated in singular. What that means is, once a plot is generated and you have viewed it, you need to close the plot for the next one to be generated and displayed.