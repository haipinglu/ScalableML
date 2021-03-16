
# Lab 7: Scalable logistic regression


## Study schedule

- [Section 1](#1-logistic-regression-in-PySpark): To finish by Wednesday. **Essential**
- [Section 2](#2-Exercises): To finish before Thursday. ***Exercise***
- [Section 3](#3-additional-exercise-optional): To explore further. *Optional*


## Introduction

In this lab, we will explore the performance of Logistic Regression on the datasets we already used in the Notebook for Decision Trees for Classification, [Lab 6](https://github.com/haipinglu/ScalableML/blob/master/Lab%206%20-%20Scalable%20Decision%20trees.md). 

We start with the [Spambase Dataset](http://archive.ics.uci.edu/ml/datasets/Spambase).

## 1. Logistic regression in PySpark

We load the dataset and the names of the features and label. We cache the dataframe for efficiently performing several operations to rawdata inside a loop.


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

# For being able to save files in a Parquet file format, later on, we need to rename
# two columns with invalid characters ; and (
spam_names[spam_names.index('char_freq_;')] = 'char_freq_semicolon'
spam_names[spam_names.index('char_freq_(')] = 'char_freq_leftp'
```

We now rename the columns using the more familiar names for the features.


```python
schemaNames = rawdata.schema.names
spam_names[ncolumns-1] = 'labels'
for i in range(ncolumns):
    rawdata = rawdata.withColumnRenamed(schemaNames[i], spam_names[i])
```

We import the Double type from pyspark.sql.types, use the withColumn method for the dataframe and cast() the column to DoubleType.


```python
from pyspark.sql.types import DoubleType
for i in range(ncolumns):
    rawdata = rawdata.withColumn(spam_names[i], rawdata[spam_names[i]].cast(DoubleType()))
```

We use the same seed that we used in the previous Notebook to split the data into training and test.


```python
(trainingDatag, testDatag) = rawdata.randomSplit([0.7, 0.3], 42)
```

**Save the training and tets sets** Once we have split the data into training and test, we can save to disk both sets so that we can use them later, for example, to compare the performance of different transformations to the data or ML models on the same training and test data. We will use the [Apache Parquet](https://en.wikipedia.org/wiki/Apache_Parquet) format to efficiently store both files.


```python
trainingDatag.write.mode("overwrite").parquet('./Data/spamdata_training.parquet')
testDatag.write.mode("overwrite").parquet('./Data/spamdata_test.parquet')
```

Let us read from disk both files


```python
trainingData = spark.read.parquet('./Data/spamdata_training.parquet')
testData = spark.read.parquet('./Data/spamdata_test.parquet')
```

We create the VectorAssembler to concatenate all the features in a vector.


```python
from pyspark.ml.feature import VectorAssembler
vecAssembler = VectorAssembler(inputCols = spam_names[0:ncolumns-1], outputCol = 'features') 
```

**Logistic regression** We are now in a position to train the logistic regression model. But before, let us look at a list of relevant parameters. A comprehensive list of parameters for [LogisticRegression](http://spark.apache.org/docs/2.3.2/api/python/pyspark.ml.html?highlight=logisticregression#pyspark.ml.classification.LogisticRegression) can be found in the Python API for PySpark.

> **maxIter**: max number of iterations. <p>
    **regParam**: regularization parameter ($\ge 0$).<p>
        **elasticNetParam**: mixing parameter for ElasticNet. It takes values in the range [0,1]. For $\alpha=0$, the penalty is an $\ell_2$. For $\alpha=1$, the penalty is an $\ell_1$.<p>
        **family**: binomial (binary classification) or multinomial (multi-class classification). It can also be 'auto'.<p>
            **standardization**: whether to standardize the training features before fitting the model. It can be true or false (True by default).
            
The function to optimise has the form
$$
f(\mathbf{w}) = LL(\mathbf{w}) + \lambda\Big[\alpha\|\mathbf{w}\|_1 + (1-\alpha)\frac{1}{2}\|\mathbf{w}\|_2\Big],
$$
where $LL(\mathbf{w})$ is the logistic loss given as
$$
LL(\mathbf{w}) = \sum_{n=1}^N \log[1+\exp(-y_n\mathbf{w}^{\top}\mathbf{x}_n)].
$$

Let us train different classifiers on the same training data. We start with logistic regression, without regularization, so $\lambda=0$.



```python
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol='features', labelCol='labels', maxIter=50, regParam=0, family="binomial")
```

We now create a pipeline for this model and fit it to the training data


```python
from pyspark.ml import Pipeline

# Combine stages into pipeline
stages = [vecAssembler, lr]
pipeline = Pipeline(stages=stages)

pipelineModel = pipeline.fit(trainingData)
```

Let us compute the accuracy.


```python
predictions = pipelineModel.transform(testData)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator\
      (labelCol="labels", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % accuracy)
```

    Accuracy = 0.921554 


We now save the vector $\mathbf{w}$ obtained without regularisation


```python
w_no_reg = pipelineModel.stages[-1].coefficients.values
```

We now train a second logistic regression classifier using only $\ell_1$ regularisation ($\lambda=0.01$ and $\alpha=1$)


```python
lrL1 = LogisticRegression(featuresCol='features', labelCol='labels', maxIter=50, regParam=0.01, \
                          elasticNetParam=1, family="binomial")

# Pipeline for the second model with L1 regularisation
stageslrL1 = [vecAssembler, lrL1]
pipelinelrL1 = Pipeline(stages=stageslrL1)
pipelineModellrL1 = pipelinelrL1.fit(trainingData)

predictions = pipelineModellrL1.transform(testData)
# With Predictions
evaluator = MulticlassClassificationEvaluator\
      (labelCol="labels", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % accuracy)
```

    Accuracy = 0.913176 


We now save the vector $\mathbf{w}$ obtained for the L1 regularisation


```python
w_L1 = pipelineModellrL1.stages[-1].coefficients.values
```

Let us plot the values of the coefficients $\mathbf{w}$ for the no regularisation case and the L1 regularisation case.


```python
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(w_no_reg)
ax1.set_title('No regularisation')
ax2.plot(w_L1)
ax2.set_title('L1 regularisation')
plt.savefig("Output/w_with_and_without_reg.png")
```

Let us find out which features are preferred by each method. Without regularisation, the most relevant feature is


```python
spam_names[np.argmax(np.abs(w_no_reg))]
```




    'word_freq_cs'



With L1 regularisation, the most relevant feature is


```python
spam_names[np.argmax(np.abs(w_L1))]
```




    'char_freq_$'



This last result is consistent with the most relevant feature given by the [Decision Tree Classifier of Lab 6](https://github.com/haipinglu/ScalableML/blob/master/Lab%206%20-%20Scalable%20Decision%20trees.md).

A useful method for the logistic regression model is the [summary](http://spark.apache.org/docs/3.0.1/api/python/pyspark.ml.html?highlight=logisticregressionsummary#pyspark.ml.classification.LogisticRegressionSummary) method. 


```python
lrModel1 = pipelineModellrL1.stages[-1]
lrModel1.summary.accuracy
```




    0.9111922141119222



The accuracy here is different to the one we got before. Why?

Other quantities that can be obtained from the summary include falsePositiveRateByLabel, precisionByLabel, recallByLabel, among others. For an exhaustive list, please read [here](http://spark.apache.org/docs/3.0.1/api/python/pyspark.ml.html?highlight=logisticregressionsummary#pyspark.ml.classification.LogisticRegressionSummary).


```python
print("Precision by label:")
for i, prec in enumerate(lrModel1.summary.precisionByLabel):
    print("label %d: %s" % (i, prec))
```

    Precision by label:
    label 0: 0.8979686057248384
    label 1: 0.9367201426024956


## 2. Exercises

**Note**: A *reference* solution will be provided in Blackboard for this part by the following Wednesday (the latest), before the quiz.


### Exercise 1

Try a pure L2 regularisation and an elastic net regularisation on the same data partitions from above. Compare accuracies and find the most relevant features for both cases. Are these features the same than the one obtained for L1 regularisation?

### Exercise 2

Instead of creating a logistic regression model trying one type of regularisation at a time, create a [ParamGridBuilder](http://spark.apache.org/docs/3.0.1/api/python/pyspark.ml.html?highlight=paramgridbuilder#pyspark.ml.tuning.ParamGridBuilder) to be used inside a [CrossValidator](http://spark.apache.org/docs/3.0.1/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator) to fine tune the best type of regularisation and the best parameters for that type of regularisation. Use five folds for the CrossValidator.

## 3. Additional exercise (optional)

**Note**: NO solutions will be provided for this part.

Create a logistic regression classifier that runs on the [default of credit cards](http://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) dataset. Several of the features in this dataset are categorical. Use the tools provided by PySpark (pyspark.ml.feature) for treating categorical variables. 

Note also that this dataset has a different format to the Spambase dataset above - you will need to convert from XLS format to, say, CSV, before using the data. You can use any available tool for this: for example, Excell has an export option, or there is a command line tool <tt>xls2csv</tt> available on Linux.
