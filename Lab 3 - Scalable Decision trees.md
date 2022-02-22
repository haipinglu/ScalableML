# Lab 3: Scalable decision trees and ensemble methods


## Study schedule

- [Section 1](#1-Decision-trees-in-PySpark): To finish in the lab session on 24th Feb. **Essential**
- [Section 2](#2-Ensemble-methods): To finish in the lab session on 24th Feb. **Essential**
- [Section 3](#3-Exercises): To finish by the following Tuesday 1st March. ***Exercise***
- [Section 4](#4-additional-exercises-optional): To explore further. *Optional*

## Introduction

In this notebook we will explore the classes in PySpark that implement decision trees, random forests and gradient-boosted trees. We will study both classification and regression. The implementations for random forests and gradient-boosted trees heavliy rely on the implementations for decision trees.

There are several challenges when implementing decision trees in a distributed setting, particularly when we want to use [commodity hardware](https://en.wikipedia.org/wiki/Commodity_computing). A very popular implementation is known as [PLANET: Massively Parallel Learning of Tree Ensembles with MapReduce](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/36296.pdf). PLANET allows an efficient implementation of decision trees at large scale purely using map() and reduce() operations, suitable for a Hadoop cluster. Although the Decision Tree implementation in [Apache Spark borrows some of the ideas from PLANET](http://jmlr.org/papers/volume17/15-237/15-237.pdf), it also introduces additional tricks that exploit the well-known advantages of Apache Spark, e.g. in memory computing. The Apache Spark implementation of the DecisionTree classifier may not be as flexible as the scikit-learn one (bear in mind they were designed under a different sets of restrictions), but it still allows the use of such a powerful machine learning model at large scale.

You can find more technical details on the implementation of Decision Trees in Apache Spark in the youtube video [Scalable Decision Trees in Spark MLlib](https://www.youtube.com/watch?v=N453EV5gHRA&t=10m30s) by Manish Amde and the youtube video [Decision Trees on Spark](https://www.youtube.com/watch?v=3WS9OK3EXVA) by Joseph Bradley. These technical details are also reviewed in a [blog post on decision trees](https://databricks.com/blog/2014/09/29/scalable-decision-trees-in-mllib.html) and another [blog post on random forests](https://databricks.com/blog/2015/01/21/random-forests-and-boosting-in-mllib.html). 

**Dependencies.** For this lab, we need to install the ``matplotlib`` and `pandas` packages. Make sure you install the packages in the environment **myspark**

Before you continue, open a new terminal in [ShARC](https://docs.hpc.shef.ac.uk/en/latest/hpc/index.html), use the `rse-com6012` queue with four nodes, and activate the **myspark** environment

`module load apps/java/jdk1.8.0_102/binary`

`module load apps/python/conda`

`source activate myspark`

You can now use pip to install the packages using

`pip install matplotlib pandas`

**You only need to install matplotlib and pandas in your environment once.** 

## 1. Decision trees in PySpark

We will build a decision tree classifier that will be able to detect spam from the text in an email. We already saw this example using [scikit-learn](https://scikit-learn.org/stable/) in the previous module [COM6509 Machine Learning and Adaptive Intelligence](https://github.com/maalvarezl/MLAI). The Notebook is in [this link](https://colab.research.google.com/github/maalvarezl/MLAI/blob/master/Labs/Lab%203%20-%20Decision%20trees%20and%20ensemble%20methods.ipynb).  

The dataset that we will use is from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php), where UCI stands for University of California Irvine. The UCI repository is and has been a valuable resource in Machine Learning. It contains datasets for classification, regression, clustering and several other machine learning problems. These datasets are open source and they have been uploaded by contributors of many research articles. 

The particular dataset that we will use wil be referred to is the [Spambase Dataset](http://archive.ics.uci.edu/ml/datasets/Spambase). A detailed description is in the previous link. The dataset contains 57 features related to word frequency, character frequency, and others related to capital letters. The description of the features and labels in the dataset is available [here](http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names). The output label indicated whether an email was considered 'ham' or 'spam', so it is a binary label. 

After installing `matplotlib` and `pandas`, go to the folder `ScalableML` in your terminal and open pyspark.

We now load the dataset and load the names of the features and label that we will use to create the schema for the dataframe. We also cache the dataframe since we are going to perform several operations to rawdata inside a loop.


```python
import numpy as np
rawdata = spark.read.csv('./Data/spambase.data')
rawdata.cache()
ncolumns = len(rawdata.columns)
spam_names = [spam_names.rstrip('\n') for spam_names in open('./Data/spambase.data.names')]
number_names = np.shape(spam_names)[0]
for i in range(number_names):
    local = spam_names[i]
    colon_pos = local.find(':')
    spam_names[i] = local[:colon_pos]
```

We use the [<tt>withColumnRenamed</tt>](https://spark.apache.org/docs/3.0.1/api/python/pyspark.sql.html?highlight=withcolumn#pyspark.sql.DataFrame.withColumnRenamed) method for the dataframe to rename the columns using the more familiar names for the features.


```python
schemaNames = rawdata.schema.names
spam_names[ncolumns-1] = 'labels'
for i in range(ncolumns):
    rawdata = rawdata.withColumnRenamed(schemaNames[i], spam_names[i])
```

Perhaps one of the most important operations when doing data analytics in Apache Spark consists in preprocessing the dataset so that it can be analysed using the MLlib package. In the case of supervised learning, classification or regression, we want the data into a column of type `Double` for the label and a column of type `SparseVector` or `DenseVector` for the features. In turn, to get this representation of the features as a vector, we first need to transform the individual features to type `Double`.

Let us see first what is the type of the original features after reading the file.


```python
rawdata.printSchema()
```

    root
     |-- word_freq_make: string (nullable = true)
     |-- word_freq_address: string (nullable = true)
     |-- word_freq_all: string (nullable = true)
     |-- word_freq_3d: string (nullable = true)
     |-- word_freq_our: string (nullable = true)
     |-- word_freq_over: string (nullable = true)
     |-- word_freq_remove: string (nullable = true)
     |-- word_freq_internet: string (nullable = true)
     |-- word_freq_order: string (nullable = true)
     |-- word_freq_mail: string (nullable = true)
     |-- word_freq_receive: string (nullable = true)
     |-- word_freq_will: string (nullable = true)
     |-- word_freq_people: string (nullable = true)
     |-- word_freq_report: string (nullable = true)
     |-- word_freq_addresses: string (nullable = true)
     |-- word_freq_free: string (nullable = true)
     |-- word_freq_business: string (nullable = true)
     |-- word_freq_email: string (nullable = true)
     |-- word_freq_you: string (nullable = true)
     |-- word_freq_credit: string (nullable = true)
     |-- word_freq_your: string (nullable = true)
     |-- word_freq_font: string (nullable = true)
     |-- word_freq_000: string (nullable = true)
     |-- word_freq_money: string (nullable = true)
     |-- word_freq_hp: string (nullable = true)
     |-- word_freq_hpl: string (nullable = true)
     |-- word_freq_george: string (nullable = true)
     |-- word_freq_650: string (nullable = true)
     |-- word_freq_lab: string (nullable = true)
     |-- word_freq_labs: string (nullable = true)
     |-- word_freq_telnet: string (nullable = true)
     |-- word_freq_857: string (nullable = true)
     |-- word_freq_data: string (nullable = true)
     |-- word_freq_415: string (nullable = true)
     |-- word_freq_85: string (nullable = true)
     |-- word_freq_technology: string (nullable = true)
     |-- word_freq_1999: string (nullable = true)
     |-- word_freq_parts: string (nullable = true)
     |-- word_freq_pm: string (nullable = true)
     |-- word_freq_direct: string (nullable = true)
     |-- word_freq_cs: string (nullable = true)
     |-- word_freq_meeting: string (nullable = true)
     |-- word_freq_original: string (nullable = true)
     |-- word_freq_project: string (nullable = true)
     |-- word_freq_re: string (nullable = true)
     |-- word_freq_edu: string (nullable = true)
     |-- word_freq_table: string (nullable = true)
     |-- word_freq_conference: string (nullable = true)
     |-- char_freq_;: string (nullable = true)
     |-- char_freq_(: string (nullable = true)
     |-- char_freq_[: string (nullable = true)
     |-- char_freq_!: string (nullable = true)
     |-- char_freq_$: string (nullable = true)
     |-- char_freq_#: string (nullable = true)
     |-- capital_run_length_average: string (nullable = true)
     |-- capital_run_length_longest: string (nullable = true)
     |-- capital_run_length_total: string (nullable = true)
     |-- labels: string (nullable = true)
    


We notice that all the features and the label are of type `String`. We import the <tt>String</tt> type from pyspark.sql.types, and later use the [<tt>withColumn</tt>](https://spark.apache.org/docs/3.0.1/api/python/pyspark.sql.html?highlight=withcolumn#pyspark.sql.DataFrame.withColumn) method for the dataframe to `cast()` each column to `Double`.


```python
from pyspark.sql.types import StringType
from pyspark.sql.functions import col

StringColumns = [x.name for x in rawdata.schema.fields if x.dataType == StringType()]
for c in StringColumns:
    rawdata = rawdata.withColumn(c, col(c).cast("double"))
```

We print the schema again and notice the variables are now of type `double`.


```python
rawdata.printSchema()
```

    root
     |-- word_freq_make: double (nullable = true)
     |-- word_freq_address: double (nullable = true)
     |-- word_freq_all: double (nullable = true)
     |-- word_freq_3d: double (nullable = true)
     |-- word_freq_our: double (nullable = true)
     |-- word_freq_over: double (nullable = true)
     |-- word_freq_remove: double (nullable = true)
     |-- word_freq_internet: double (nullable = true)
     |-- word_freq_order: double (nullable = true)
     |-- word_freq_mail: double (nullable = true)
     |-- word_freq_receive: double (nullable = true)
     |-- word_freq_will: double (nullable = true)
     |-- word_freq_people: double (nullable = true)
     |-- word_freq_report: double (nullable = true)
     |-- word_freq_addresses: double (nullable = true)
     |-- word_freq_free: double (nullable = true)
     |-- word_freq_business: double (nullable = true)
     |-- word_freq_email: double (nullable = true)
     |-- word_freq_you: double (nullable = true)
     |-- word_freq_credit: double (nullable = true)
     |-- word_freq_your: double (nullable = true)
     |-- word_freq_font: double (nullable = true)
     |-- word_freq_000: double (nullable = true)
     |-- word_freq_money: double (nullable = true)
     |-- word_freq_hp: double (nullable = true)
     |-- word_freq_hpl: double (nullable = true)
     |-- word_freq_george: double (nullable = true)
     |-- word_freq_650: double (nullable = true)
     |-- word_freq_lab: double (nullable = true)
     |-- word_freq_labs: double (nullable = true)
     |-- word_freq_telnet: double (nullable = true)
     |-- word_freq_857: double (nullable = true)
     |-- word_freq_data: double (nullable = true)
     |-- word_freq_415: double (nullable = true)
     |-- word_freq_85: double (nullable = true)
     |-- word_freq_technology: double (nullable = true)
     |-- word_freq_1999: double (nullable = true)
     |-- word_freq_parts: double (nullable = true)
     |-- word_freq_pm: double (nullable = true)
     |-- word_freq_direct: double (nullable = true)
     |-- word_freq_cs: double (nullable = true)
     |-- word_freq_meeting: double (nullable = true)
     |-- word_freq_original: double (nullable = true)
     |-- word_freq_project: double (nullable = true)
     |-- word_freq_re: double (nullable = true)
     |-- word_freq_edu: double (nullable = true)
     |-- word_freq_table: double (nullable = true)
     |-- word_freq_conference: double (nullable = true)
     |-- char_freq_;: double (nullable = true)
     |-- char_freq_(: double (nullable = true)
     |-- char_freq_[: double (nullable = true)
     |-- char_freq_!: double (nullable = true)
     |-- char_freq_$: double (nullable = true)
     |-- char_freq_#: double (nullable = true)
     |-- capital_run_length_average: double (nullable = true)
     |-- capital_run_length_longest: double (nullable = true)
     |-- capital_run_length_total: double (nullable = true)
     |-- labels: double (nullable = true)
    


We have now a dataframe that contains several columns corresponding to the features, of type double, and the last column corresponding to the labels, also of type double. 

We can now start the machine learning analysis by creating the training and test set and then designing the DecisionTreeClassifier using the training data.


```python
trainingData, testData = rawdata.randomSplit([0.7, 0.3], 1242)
print(f"There are {trainingData.cache().count()} rows in the training set, and {testData.cache().count()} in the test set")
```

    There are 3166 rows in the training set, and 1435 in the test set


**Be careful when using randomSplit**. It is important to notice that, unlike when using a single machine, the `randomSplit` method can lead to different training and test sets **even if we use the same seed!** This can happen when the cluster configuration changes. The dataset we are using is very small, it basically fits in one partition. We can see the effect of cluster configuration by repartitioning the dataframe. 


```python
trainRepartitionData, testRepartitionData = (rawdata.repartition(24).randomSplit([0.7, 0.3], seed=1242))
print(trainRepartitionData.count())
```

    3223


When you do a 70/30 train/test split, it is an "approximate" 70/30 split. It is not an exact 70/30 split, and when the partitioning of the data changes, you get not only a different number of data points in train/test, but also different data points.

The recommendation is to split your data once. If you need to modify the training and test data in some way, make sure you perform the modifications on the training and test data you got from the split you did at the beginning. Do not perform the modifications on the original data and perform again randomSplit with the hope you will get the same training and test splits that you got the first time.

**Vector Assembler** Most supervised learning models in PySpark require a column of type `SparseVector` or `DenseVector` for the features. We use the [<tt>VectorAssembler</tt>](http://spark.apache.org/docs/3.0.1/api/python/pyspark.ml.html?highlight=vectorassembler#pyspark.ml.feature.VectorAssembler) tool to concatenate all the features in a vector.


```python
from pyspark.ml.feature import VectorAssembler
vecAssembler = VectorAssembler(inputCols = spam_names[0:ncolumns-1], outputCol = 'features') 
vecTrainingData = vecAssembler.transform(trainingData)
vecTrainingData.select("features", "labels").show(5)
```

    +--------------------+------+
    |            features|labels|
    +--------------------+------+
    |(57,[54,55,56],[1...|   0.0|
    |(57,[54,55,56],[1...|   0.0|
    |(57,[54,55,56],[1...|   0.0|
    |(57,[54,55,56],[1...|   0.0|
    |(57,[54,55,56],[1...|   0.0|
    +--------------------+------+
    only showing top 5 rows
    


The [DecisionTreeClassifier](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.DecisionTreeClassifier.html#pyspark.ml.classification.DecisionTreeClassifier) implemented in PySpark has several parameters to tune. Some of them are

> **maxDepth**: it corresponds to the maximum depth of the tree. The default is 5.<p>
**maxBins**: it determines how many bins should be created from continuous features. The default is 32.<p>
    **impurity**: it is the metric used to compute information gain. The options are "gini" or "entropy". The default is "gini".<p>
        **minInfoGain**: it determines the minimum information gain that will be used for a split. The default is zero.



```python
from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(labelCol="labels", featuresCol="features", maxDepth=10, impurity='entropy')
model = dt.fit(vecTrainingData)
```

The individual importance of the features can be obtained using [featureImportances](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.DecisionTreeClassificationModel.html?highlight=featureimportance#pyspark.ml.classification.DecisionTreeClassificationModel.featureImportances). We can recover the values of importance and the indexes of the vector where these values are different from zero using values and indices


```python
fi = model.featureImportances
imp_feat = np.zeros(ncolumns-1)
imp_feat[fi.indices] = fi.values
```

Let us plot the relative importances. 

Although we can not visualise an image on the HPC terminal, we can print a figure as a .png file and visualise the .png file in the screen in the local computer.

To plot and save figures on HPC, we need to do the following before using pyplot:


```python
import matplotlib 
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab! 
```

We now set the instructions to print the figure in the `Output` folder


```python
import matplotlib.pyplot as plt
x = np.arange(ncolumns-1)
plt.bar(x, imp_feat)
plt.savefig("./Output/feature_importances.png")
```

The feature with the highest importance is 


```python
spam_names[np.argmax(imp_feat)]
```




    'char_freq_$'



We can visualise the DecisionTree in the form of *if-then-else* statements.


```python
print(model.toDebugString)
```

    DecisionTreeClassificationModel: uid=DecisionTreeClassifier_7144f32ac0ce, depth=10, numNodes=193, numClasses=2, numFeatures=57
      If (feature 52 <= 0.0455)
       If (feature 6 <= 0.045)
        If (feature 51 <= 0.172)
         If (feature 24 <= 0.025)
          If (feature 55 <= 10.5)
           If (feature 20 <= 0.615)
            If (feature 26 <= 0.005)
             If (feature 9 <= 0.265)
              Predict: 0.0
             Else (feature 9 > 0.265)
              If (feature 56 <= 28.5)
               Predict: 0.0
              Else (feature 56 > 28.5)
               If (feature 2 <= 0.015)
                Predict: 1.0
               Else (feature 2 > 0.015)
                Predict: 0.0
            Else (feature 26 > 0.005)
             Predict: 0.0
           Else (feature 20 > 0.615)
            If (feature 15 <= 1.8050000000000002)
             If (feature 7 <= 0.01)
              If (feature 35 <= 0.005)
               If (feature 19 <= 0.01)
                Predict: 0.0
               Else (feature 19 > 0.01)
                Predict: 1.0
              Else (feature 35 > 0.005)
               If (feature 2 <= 0.655)
                Predict: 1.0
               Else (feature 2 > 0.655)
                Predict: 0.0
             Else (feature 7 > 0.01)
              If (feature 26 <= 0.005)
               Predict: 1.0
              Else (feature 26 > 0.005)
               Predict: 0.0
            Else (feature 15 > 1.8050000000000002)
             If (feature 40 <= 0.02)
              Predict: 1.0
             Else (feature 40 > 0.02)
              Predict: 0.0
          Else (feature 55 > 10.5)
           If (feature 44 <= 0.265)
            If (feature 26 <= 0.005)
             If (feature 4 <= 0.675)
              If (feature 45 <= 0.01)
               Predict: 0.0
              Else (feature 45 > 0.01)
               If (feature 27 <= 0.045)
                Predict: 0.0
               Else (feature 27 > 0.045)
                Predict: 1.0
             Else (feature 4 > 0.675)
              If (feature 51 <= 0.0335)
               Predict: 1.0
              Else (feature 51 > 0.0335)
               If (feature 11 <= 0.895)
                Predict: 1.0
               Else (feature 11 > 0.895)
                Predict: 0.0
            Else (feature 26 > 0.005)
             Predict: 0.0
           Else (feature 44 > 0.265)
            If (feature 15 <= 1.085)
             Predict: 0.0
            Else (feature 15 > 1.085)
             If (feature 1 <= 0.005)
              Predict: 1.0
             Else (feature 1 > 0.005)
              Predict: 0.0
         Else (feature 24 > 0.025)
          If (feature 15 <= 1.8050000000000002)
           If (feature 22 <= 0.005)
            Predict: 0.0
           Else (feature 22 > 0.005)
            If (feature 36 <= 0.805)
             Predict: 0.0
            Else (feature 36 > 0.805)
             Predict: 1.0
          Else (feature 15 > 1.8050000000000002)
           Predict: 1.0
        Else (feature 51 > 0.172)
         If (feature 54 <= 2.6285)
          If (feature 15 <= 0.01)
           If (feature 7 <= 0.01)
            If (feature 16 <= 0.01)
             If (feature 51 <= 0.976)
              If (feature 8 <= 0.005)
               If (feature 22 <= 0.365)
                Predict: 0.0
               Else (feature 22 > 0.365)
                Predict: 1.0
              Else (feature 8 > 0.005)
               If (feature 55 <= 16.5)
                Predict: 0.0
               Else (feature 55 > 16.5)
                Predict: 1.0
             Else (feature 51 > 0.976)
              If (feature 44 <= 0.005)
               If (feature 54 <= 1.6795)
                Predict: 1.0
               Else (feature 54 > 1.6795)
                Predict: 0.0
              Else (feature 44 > 0.005)
               Predict: 0.0
            Else (feature 16 > 0.01)
             If (feature 24 <= 0.01)
              Predict: 1.0
             Else (feature 24 > 0.01)
              Predict: 0.0
           Else (feature 7 > 0.01)
            Predict: 1.0
          Else (feature 15 > 0.01)
           If (feature 45 <= 0.01)
            If (feature 51 <= 0.4955)
             If (feature 15 <= 1.8050000000000002)
              If (feature 2 <= 0.755)
               If (feature 43 <= 0.005)
                Predict: 1.0
               Else (feature 43 > 0.005)
                Predict: 0.0
              Else (feature 2 > 0.755)
               Predict: 0.0
             Else (feature 15 > 1.8050000000000002)
              Predict: 1.0
            Else (feature 51 > 0.4955)
             If (feature 56 <= 40.5)
              If (feature 15 <= 0.775)
               Predict: 0.0
              Else (feature 15 > 0.775)
               Predict: 1.0
             Else (feature 56 > 40.5)
              Predict: 1.0
           Else (feature 45 > 0.01)
            Predict: 0.0
         Else (feature 54 > 2.6285)
          If (feature 51 <= 0.4955)
           If (feature 34 <= 0.005)
            If (feature 11 <= 2.005)
             If (feature 50 <= 0.001)
              If (feature 20 <= 0.885)
               If (feature 15 <= 1.085)
                Predict: 0.0
               Else (feature 15 > 1.085)
                Predict: 1.0
              Else (feature 20 > 0.885)
               If (feature 49 <= 0.3795)
                Predict: 1.0
               Else (feature 49 > 0.3795)
                Predict: 0.0
             Else (feature 50 > 0.001)
              Predict: 1.0
            Else (feature 11 > 2.005)
             Predict: 0.0
           Else (feature 34 > 0.005)
            Predict: 0.0
          Else (feature 51 > 0.4955)
           If (feature 49 <= 0.002)
            If (feature 44 <= 0.005)
             If (feature 56 <= 17.5)
              If (feature 18 <= 0.01)
               Predict: 0.0
              Else (feature 18 > 0.01)
               Predict: 1.0
             Else (feature 56 > 17.5)
              If (feature 45 <= 0.01)
               Predict: 1.0
              Else (feature 45 > 0.01)
               Predict: 0.0
            Else (feature 44 > 0.005)
             If (feature 44 <= 2.1)
              Predict: 0.0
             Else (feature 44 > 2.1)
              Predict: 1.0
           Else (feature 49 > 0.002)
            Predict: 1.0
       Else (feature 6 > 0.045)
        If (feature 26 <= 0.005)
         If (feature 29 <= 0.185)
          If (feature 51 <= 0.001)
           If (feature 6 <= 0.405)
            If (feature 4 <= 0.10500000000000001)
             Predict: 0.0
            Else (feature 4 > 0.10500000000000001)
             If (feature 0 <= 0.175)
              Predict: 1.0
             Else (feature 0 > 0.175)
              Predict: 0.0
           Else (feature 6 > 0.405)
            If (feature 32 <= 0.01)
             If (feature 9 <= 1.125)
              Predict: 1.0
             Else (feature 9 > 1.125)
              If (feature 0 <= 0.005)
               Predict: 0.0
              Else (feature 0 > 0.005)
               Predict: 1.0
            Else (feature 32 > 0.01)
             Predict: 0.0
          Else (feature 51 > 0.001)
           If (feature 45 <= 0.01)
            If (feature 56 <= 46.5)
             If (feature 54 <= 2.498)
              Predict: 1.0
             Else (feature 54 > 2.498)
              If (feature 55 <= 12.5)
               Predict: 0.0
              Else (feature 55 > 12.5)
               Predict: 1.0
            Else (feature 56 > 46.5)
             Predict: 1.0
           Else (feature 45 > 0.01)
            If (feature 2 <= 0.015)
             Predict: 0.0
            Else (feature 2 > 0.015)
             Predict: 1.0
         Else (feature 29 > 0.185)
          Predict: 0.0
        Else (feature 26 > 0.005)
         Predict: 0.0
      Else (feature 52 > 0.0455)
       If (feature 24 <= 0.195)
        If (feature 55 <= 9.5)
         If (feature 15 <= 0.165)
          If (feature 18 <= 3.1550000000000002)
           Predict: 0.0
          Else (feature 18 > 3.1550000000000002)
           Predict: 1.0
         Else (feature 15 > 0.165)
          If (feature 55 <= 8.5)
           If (feature 12 <= 0.805)
            If (feature 4 <= 0.825)
             Predict: 1.0
            Else (feature 4 > 0.825)
             If (feature 1 <= 0.005)
              Predict: 0.0
             Else (feature 1 > 0.005)
              Predict: 1.0
           Else (feature 12 > 0.805)
            Predict: 0.0
          Else (feature 55 > 8.5)
           Predict: 0.0
        Else (feature 55 > 9.5)
         If (feature 45 <= 0.645)
          If (feature 22 <= 0.365)
           If (feature 51 <= 0.1325)
            If (feature 49 <= 0.6515)
             If (feature 54 <= 3.4275)
              If (feature 12 <= 0.01)
               If (feature 1 <= 0.135)
                Predict: 1.0
               Else (feature 1 > 0.135)
                Predict: 0.0
              Else (feature 12 > 0.01)
               Predict: 1.0
             Else (feature 54 > 3.4275)
              Predict: 1.0
            Else (feature 49 > 0.6515)
             Predict: 0.0
           Else (feature 51 > 0.1325)
            If (feature 36 <= 0.14500000000000002)
             If (feature 54 <= 1.598)
              Predict: 0.0
             Else (feature 54 > 1.598)
              Predict: 1.0
            Else (feature 36 > 0.14500000000000002)
             If (feature 18 <= 1.7650000000000001)
              Predict: 1.0
             Else (feature 18 > 1.7650000000000001)
              If (feature 1 <= 0.225)
               Predict: 0.0
              Else (feature 1 > 0.225)
               Predict: 1.0
          Else (feature 22 > 0.365)
           Predict: 1.0
         Else (feature 45 > 0.645)
          Predict: 0.0
       Else (feature 24 > 0.195)
        If (feature 51 <= 0.4325)
         If (feature 6 <= 0.215)
          Predict: 0.0
         Else (feature 6 > 0.215)
          If (feature 1 <= 0.005)
           Predict: 1.0
          Else (feature 1 > 0.005)
           Predict: 0.0
        Else (feature 51 > 0.4325)
         Predict: 1.0
    


Indirectly, decision trees allow feature selection: features that allow making decisions in the top of the tree are more relevant for the decision problem.

We can organise the information provided by the visualisation above in the form of a Table using Pandas


```python
import pandas as pd
featureImp = pd.DataFrame(
  list(zip(vecAssembler.getInputCols(), model.featureImportances)),
  columns=["feature", "importance"])
featureImp.sort_values(by="importance", ascending=False)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>52</th>
      <td>char_freq_$</td>
      <td>0.289479</td>
    </tr>
    <tr>
      <th>6</th>
      <td>word_freq_remove</td>
      <td>0.150070</td>
    </tr>
    <tr>
      <th>51</th>
      <td>char_freq_!</td>
      <td>0.121890</td>
    </tr>
    <tr>
      <th>24</th>
      <td>word_freq_hp</td>
      <td>0.084602</td>
    </tr>
    <tr>
      <th>55</th>
      <td>capital_run_length_longest</td>
      <td>0.049771</td>
    </tr>
    <tr>
      <th>54</th>
      <td>capital_run_length_average</td>
      <td>0.042403</td>
    </tr>
    <tr>
      <th>15</th>
      <td>word_freq_free</td>
      <td>0.036760</td>
    </tr>
    <tr>
      <th>45</th>
      <td>word_freq_edu</td>
      <td>0.031195</td>
    </tr>
    <tr>
      <th>26</th>
      <td>word_freq_george</td>
      <td>0.030779</td>
    </tr>
    <tr>
      <th>44</th>
      <td>word_freq_re</td>
      <td>0.018478</td>
    </tr>
    <tr>
      <th>4</th>
      <td>word_freq_our</td>
      <td>0.011667</td>
    </tr>
    <tr>
      <th>20</th>
      <td>word_freq_your</td>
      <td>0.011351</td>
    </tr>
    <tr>
      <th>7</th>
      <td>word_freq_internet</td>
      <td>0.010919</td>
    </tr>
    <tr>
      <th>22</th>
      <td>word_freq_000</td>
      <td>0.010024</td>
    </tr>
    <tr>
      <th>27</th>
      <td>word_freq_650</td>
      <td>0.009521</td>
    </tr>
    <tr>
      <th>49</th>
      <td>char_freq_(</td>
      <td>0.009478</td>
    </tr>
    <tr>
      <th>18</th>
      <td>word_freq_you</td>
      <td>0.007694</td>
    </tr>
    <tr>
      <th>12</th>
      <td>word_freq_people</td>
      <td>0.007015</td>
    </tr>
    <tr>
      <th>16</th>
      <td>word_freq_business</td>
      <td>0.006899</td>
    </tr>
    <tr>
      <th>1</th>
      <td>word_freq_address</td>
      <td>0.006322</td>
    </tr>
    <tr>
      <th>2</th>
      <td>word_freq_all</td>
      <td>0.006321</td>
    </tr>
    <tr>
      <th>36</th>
      <td>word_freq_1999</td>
      <td>0.005759</td>
    </tr>
    <tr>
      <th>11</th>
      <td>word_freq_will</td>
      <td>0.005730</td>
    </tr>
    <tr>
      <th>34</th>
      <td>word_freq_85</td>
      <td>0.005229</td>
    </tr>
    <tr>
      <th>56</th>
      <td>capital_run_length_total</td>
      <td>0.005106</td>
    </tr>
    <tr>
      <th>8</th>
      <td>word_freq_order</td>
      <td>0.004539</td>
    </tr>
    <tr>
      <th>9</th>
      <td>word_freq_mail</td>
      <td>0.004016</td>
    </tr>
    <tr>
      <th>50</th>
      <td>char_freq_[</td>
      <td>0.003810</td>
    </tr>
    <tr>
      <th>29</th>
      <td>word_freq_labs</td>
      <td>0.003350</td>
    </tr>
    <tr>
      <th>0</th>
      <td>word_freq_make</td>
      <td>0.002008</td>
    </tr>
    <tr>
      <th>35</th>
      <td>word_freq_technology</td>
      <td>0.001775</td>
    </tr>
    <tr>
      <th>32</th>
      <td>word_freq_data</td>
      <td>0.001637</td>
    </tr>
    <tr>
      <th>40</th>
      <td>word_freq_cs</td>
      <td>0.001493</td>
    </tr>
    <tr>
      <th>19</th>
      <td>word_freq_credit</td>
      <td>0.001481</td>
    </tr>
    <tr>
      <th>43</th>
      <td>word_freq_project</td>
      <td>0.001430</td>
    </tr>
    <tr>
      <th>3</th>
      <td>word_freq_3d</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>word_freq_over</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>53</th>
      <td>char_freq_#</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>word_freq_receive</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>word_freq_report</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>word_freq_addresses</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>word_freq_email</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>48</th>
      <td>char_freq_;</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>47</th>
      <td>word_freq_conference</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>46</th>
      <td>word_freq_table</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>30</th>
      <td>word_freq_telnet</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>word_freq_font</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>31</th>
      <td>word_freq_857</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>42</th>
      <td>word_freq_original</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>41</th>
      <td>word_freq_meeting</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>39</th>
      <td>word_freq_direct</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>38</th>
      <td>word_freq_pm</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>37</th>
      <td>word_freq_parts</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>word_freq_money</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>word_freq_hpl</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>33</th>
      <td>word_freq_415</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>word_freq_lab</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



A better visualisation of the tree in pyspark can be obtained by using, for example, [spark-tree-plotting](https://github.com/julioasotodv/spark-tree-plotting). The trick is to convert the spark tree to a JSON format. Once you have the JSON format, you can visualise it using [D3](https://d3js.org/) or you can transform from JSON to DOT and use graphviz as we did in scickit-learn for the Notebook in MLAI.

**Pipeline** We have not mentioned the test data yet. Before applying the decision tree to the test data, this is a good opportunity to introduce a pipeline that includes the VectorAssembler and the Decision Tree.


```python
from pyspark.ml import Pipeline

# Combine stages into pipeline
stages = [vecAssembler, dt]
pipeline = Pipeline(stages=stages)

pipelineModel = pipeline.fit(trainingData)
```

We finally use the [MulticlassClassificationEvaluator](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.MulticlassClassificationEvaluator.html?highlight=multiclassclassificationevaluator#pyspark.ml.evaluation.MulticlassClassificationEvaluator) tool to assess the accuracy on the test set.


```python
predictions = pipelineModel.transform(testData)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator\
      (labelCol="labels", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % accuracy)
```

    Accuracy = 0.915679 


### Decision trees for regression

The main difference between Decision Trees for Classification and Decision Trees for Regression is in the impurity measure used. For regression, PySpark uses the variance of the target features as the impurity measure. 

The [DecisionTreeRegressor](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.DecisionTreeRegressor.html?highlight=decisiontreeregressor) implemented in PySpark has several parameters to tune. Some of them are

> **maxDepth**: it corresponds to the maximum depth of the tree. The default is 5.<p>
**maxBins**: it determines how many bins should be created from continuous features. The default is 32.<p>
    **impurity**: for regression the only supported impurity option is variance.<p>
        **minInfoGain**: it determines the minimum information gain that will be used for a split. The default is zero.

You will have the opportunity to experiment with the [DecisionTreeRegressor](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.DecisionTreeRegressor.html?highlight=decisiontreeregressor) class later in the module.

## 2. Ensemble methods

We studied the implementation of ensemble methods in scikit-learn in COM6509. See [this notebook](https://colab.research.google.com/github/maalvarezl/MLAI/blob/master/Labs/Lab%203%20-%20Decision%20trees%20and%20ensemble%20methods.ipynb) for a refresher.   

PySpark implemenst two types of Tree Ensembles, random forests and gradient boosting. The main difference between both methods is the way in which they combine the different trees that compose the ensemble.

### Random Forests

The variant of Random Forests implemented in Apache Spark is also known as bagging or boostrap aggregating. The tree ensemble in random forests is built by training individual decision trees on different subsets of the training data and using a subset of the available features. For classification, the prediction is done by majority voting among the individual trees. For regression, the prediction is the average of the individual predictions of each tree. For more details on the PySpark implementation see [here](https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-classifier). 

Besides the parameters that we already mentioned for the [DecisionTreeClassifier](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.DecisionTreeClassifier.html#pyspark.ml.classification.DecisionTreeClassifier) and the [DecisionTreeRegressor](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.DecisionTreeRegressor.html), the [RandomForestClassifier](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.RandomForestClassifier.html) and the [RandomForestRegressor](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.RandomForestRegressor.html) in PySpark require three additional parameters:
> **numTrees** the total number of trees to train<p>
**featureSubsetStrategy** number of features to use as candidates for splitting at each tree node. Options include all, onethird, sqrt, log2, [1-n]<p>
    **subsamplingRate**: size of the dataset used for training each tree in the forest, as a fraction of the size of the original dataset. 

We already did an example of classification with decision trees. Let us use now random forests for performing regression.

#### Predicting the quality of wine

We are going to use the [Wine Quality Dataset](http://archive.ics.uci.edu/ml/datasets/Wine+Quality) to illustrate the use of the **RandomForestRegressor** class in PySpark. There are eleven input features corresponding to different attributes measured on wine samples (based on physicochemical tests). The target feature corresponds to a quality index that goes from zero to ten being zero a *very bad* wine and ten an *excellent* wine. The target feature was computed as the median score of three independent wine taster experts. More details on the dataset can be found in this [paper](https://www.sciencedirect.com/science/article/pii/S0167923609001377). 


```python
rawdataw = spark.read.csv('./Data/winequality-white.csv', sep=';', header='true')
rawdataw.cache()
```




    DataFrame[fixed acidity: string, volatile acidity: string, citric acid: string, residual sugar: string, chlorides: string, free sulfur dioxide: string, total sulfur dioxide: string, density: string, pH: string, sulphates: string, alcohol: string, quality: string]



Notice that we use the parameter `sep=;` when loading the file, since the columns in the file are separated by `;` instead of the default `,`


```python
rawdataw.printSchema()
```

    root
     |-- fixed acidity: string (nullable = true)
     |-- volatile acidity: string (nullable = true)
     |-- citric acid: string (nullable = true)
     |-- residual sugar: string (nullable = true)
     |-- chlorides: string (nullable = true)
     |-- free sulfur dioxide: string (nullable = true)
     |-- total sulfur dioxide: string (nullable = true)
     |-- density: string (nullable = true)
     |-- pH: string (nullable = true)
     |-- sulphates: string (nullable = true)
     |-- alcohol: string (nullable = true)
     |-- quality: string (nullable = true)
    


We now follow a very familiar procedure to get the dataset to a format that can be input to Spark MLlib, which consists of:
1. transforming the data from type string to type double.
2. creating a pipeline that includes a vector assembler and a random forest regressor.

We first start transforming the data types.


```python
from pyspark.sql.types import StringType
from pyspark.sql.functions import col

StringColumns = [x.name for x in rawdataw.schema.fields if x.dataType == StringType()]
for c in StringColumns:
    rawdataw = rawdataw.withColumn(c, col(c).cast("double"))
rawdataw = rawdataw.withColumnRenamed('quality', 'labels')    
```

Notice that we used the withColumnRenamed method to rename the name of the target feature from 'quality' to 'label'.


```python
rawdataw.printSchema()
```

    root
     |-- fixed acidity: double (nullable = true)
     |-- volatile acidity: double (nullable = true)
     |-- citric acid: double (nullable = true)
     |-- residual sugar: double (nullable = true)
     |-- chlorides: double (nullable = true)
     |-- free sulfur dioxide: double (nullable = true)
     |-- total sulfur dioxide: double (nullable = true)
     |-- density: double (nullable = true)
     |-- pH: double (nullable = true)
     |-- sulphates: double (nullable = true)
     |-- alcohol: double (nullable = true)
     |-- labels: double (nullable = true)
    


We now partition the data into a training and a test set


```python
trainingDataw, testDataw = rawdataw.randomSplit([0.7, 0.3], 42)
```

Now, we create the pipeline. First, we create the vector assembler.


```python
vecAssemblerw = VectorAssembler(inputCols=StringColumns[:-1], outputCol="features")
```

And now, the Random Forests regressor and the pipeline


```python
from pyspark.ml.regression import RandomForestRegressor
rf = RandomForestRegressor(labelCol="labels", featuresCol="features", maxDepth=5, numTrees=3, \
                           featureSubsetStrategy = 'all', seed=123, bootstrap=False)

stages = [vecAssemblerw, rf]
pipeline = Pipeline(stages=stages)

pipelineModelw = pipeline.fit(trainingDataw)
```

We apply now the pipeline to the test data and compute the RMSE between the predictions and the ground truth


```python
predictions = pipelineModelw.transform(testDataw)

from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator\
      (labelCol="labels", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("RMSE = %g " % rmse)
```

    RMSE = 0.755316 



```python
pipelineModelw.stages[-1].featureImportances
```




    SparseVector(11, {0: 0.0378, 1: 0.2075, 2: 0.0354, 3: 0.0266, 5: 0.1073, 7: 0.0394, 8: 0.0189, 9: 0.0106, 10: 0.5166})




```python
featureImp = pd.DataFrame(
  list(zip(vecAssemblerw.getInputCols(), pipelineModelw.stages[-1].featureImportances)),
  columns=["feature", "importance"])
featureImp.sort_values(by="importance", ascending=False)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>alcohol</td>
      <td>0.516600</td>
    </tr>
    <tr>
      <th>1</th>
      <td>volatile acidity</td>
      <td>0.207466</td>
    </tr>
    <tr>
      <th>5</th>
      <td>free sulfur dioxide</td>
      <td>0.107308</td>
    </tr>
    <tr>
      <th>7</th>
      <td>density</td>
      <td>0.039419</td>
    </tr>
    <tr>
      <th>0</th>
      <td>fixed acidity</td>
      <td>0.037760</td>
    </tr>
    <tr>
      <th>2</th>
      <td>citric acid</td>
      <td>0.035437</td>
    </tr>
    <tr>
      <th>3</th>
      <td>residual sugar</td>
      <td>0.026573</td>
    </tr>
    <tr>
      <th>8</th>
      <td>pH</td>
      <td>0.018871</td>
    </tr>
    <tr>
      <th>9</th>
      <td>sulphates</td>
      <td>0.010567</td>
    </tr>
    <tr>
      <th>4</th>
      <td>chlorides</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>total sulfur dioxide</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Gradient Boosting

In [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) or [Gradient-boosted trees](https://en.wikipedia.org/wiki/Gradient_boosting#Gradient_tree_boosting) (GBT), each tree in the ensemble is trained sequentially: the first tree is trained as usual using the training data, the second tree is trained on the residuals between the predictions of the first tree and the labels of the training data, the third tree is trained on the residuals of the predictions of the second tree, etc. The predictions of the ensemble will be the sum of the predictions of each individual tree. The type of residuals are related to the loss function that wants to be minimised. In the PySpark implementations of Gradient-Boosted trees, the loss function for binary classification is the Log-Loss function and the loss function for regression is either the squared error or the absolute error. For details, follow this [link](https://spark.apache.org/docs/latest/mllib-ensembles.html#gradient-boosted-trees-gbts).  

PySpark uses the classes [GBTRegressor](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.GBTRegressor.html) for the implementation of Gradient-Boosted trees for regression and [GBTClassifier](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.GBTClassifier.html) for the implementation of Gradient-Boosted trees for binary classification. As of PySpark version 3.0.1, GBT have not been implemented for multiclass classification.

Besides the parameters that can be specified for Decision Trees, both classes share the additional following parameters

>**lossType** type of loss function. Options are "squared" and "absolute" for regression and "logistic" for classification. <p>
    **maxIter** number of trees in the ensemble. Each iteration produces one tree.<p>
    **stepSize** also known as the learning rate, it is used for shrinking the contribution of each tree in the sequence. The default is 0.1<p>
    **subsamplingRate** as it was the case for Random Forest, this parameter is used for specifying the fraction of the training data used for learning each decision tree.    

We will now use the GBTRegressor on the wine quality dataset.


```python
from pyspark.ml.regression import GBTRegressor
gbt = GBTRegressor(labelCol="labels", featuresCol="features", \
                   maxDepth=5, maxIter=5, lossType='squared', subsamplingRate= 0.5, seed=34)

# Create the pipeline
stages = [vecAssemblerw, gbt]
pipeline = Pipeline(stages=stages)
pipelineModelg = pipeline.fit(trainingDataw)

# Apply the pipeline to the test data
predictions = pipelineModelg.transform(testDataw)

from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator \
      (labelCol="labels", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("RMSE = %g " % rmse)
```

    RMSE = 0.758747 


## 3. Exercises

**Note**: A *reference* solution will be provided in Blackboard for this part by the following Wednesday (the latest).

### Exercise 1

Include a cross-validation step for the pipeline of the [decision tree classifier applied to the spambase dataset](#1-Decision-trees-in-PySpark). An example of a cross-validator can be found [here](https://spark.apache.org/docs/latest/ml-tuning.html#cross-validation). Make <tt>paramGrid</tt> contains different values for <tt>maxDepth</tt>, <tt>maxBins</tt> and <tt>impurity</tt> and find the best parameters and associated accuracy on the test data.
1
### Exercise 2

Apply a [RandomForestClassifier](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.RandomForestClassifier.html) to the spambase dataset. As in exercise 1, include a cross-validation step with a paramGrid with options for <tt>maxDepth</tt>, <tt>maxBins</tt>, <tt>numTrees</tt>, <tt>featureSubsetStrategy</tt> and <tt>subsamplingRate</tt>. Find the best parameters and associated accuracy on the test data.

### Exercise 3

As we did for the Decision Trees for Classification, it is possible to use the [featureImportances](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.RandomForestRegressionModel.html?highlight=featureimportances#pyspark.ml.regression.RandomForestRegressionModel.featureImportances) method to study the relative importance of each feature in random forests. Use the *featureImportances* in the random forest regressor used for the wine dataset and indicate the three most relevant features. How are the feature importances computed? 

## 4. Additional exercises (optional)

**Note**: NO solutions will be provided for this part.

### Additional exercise 1 

Create a decision tree classifier that runs on the [default of credit cards](http://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) dataset. Several of the features in this dataset are categorical. Use [StringIndexer](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.StringIndexer.html) for treating the categorical variables. 

Note also that this dataset has a different format to the Spambase dataset above - you will need to convert from XLS format to, say, CSV, before using the data. You can use any available tool for this: for example, Excel has an export option, or there is a command line tool <tt>xls2csv</tt> available on Linux.

### Additional exercise 2

Write and run an HPC standalone program using random forest regression on the [Physical Activity Monitoring](http://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring) dataset, methodically experimenting with the parameters <tt>maxDepth</tt>, <tt>numTrees</tt> and <tt>subsamplingRate</tt>. Obtain the timing for the experiment. Note that the <tt>physical activity monitoring</tt> dataset contains <tt>NaN</tt> (not a number) values when values are missing - you should try dealing with this in two ways

1. Drop lines containing <tt>NaN</tt>
2. Replace <tt>NaN</tt> with the average value from that column. For this, you can use the [Imputer](http://spark.apache.org/docs/latest/ml-features.html#imputer) transformer available in <tt>pyspark.ml.feature</tt> 

Run experiments with both options.
