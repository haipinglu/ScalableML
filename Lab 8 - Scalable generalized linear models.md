
# Lab 8: Scalable Generalized Linear Models 


## Study schedule

- [Section 1](#1-glms-in-PySpark): To finish by Wednesday. **Essential**
- [Section 2](#2-Exercises): To finish before the next Monday. ***Exercise***
- [Section 3](#3-additional-exercise-optional): To explore further. *Optional*

## Introduction

Unlike linear regression, where the output is assumed to follow a Gaussian distribution, 
in [generalized linear models](https://en.wikipedia.org/wiki/Generalized_linear_model) (GLMs) the response variable $y$ follows some distribution from the [exponential family of distributions](https://en.wikipedia.org/wiki/Exponential_family).

## 1. GLMs in PySpark

In this Lab, we will look at Poisson regression over the [Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset). We will compare the performance of several models and algorithms on this dataset, including: Poisson Regression, Linear Regression implemented with IRLS and Linear Regression with general regularisation. 

We load the data. From the available .csv files, we use the hourly recordings only. 


```python
rawdata = spark.read.csv('./Data/hour.csv', header=True)
rawdata.cache()
```




    DataFrame[instant: string, dteday: string, season: string, yr: string, mnth: string, hr: string, holiday: string, weekday: string, workingday: string, weathersit: string, temp: string, atemp: string, hum: string, windspeed: string, casual: string, registered: string, cnt: string]



The following is a description of the features

    - instant: record index
	- dteday : date
	- season : season (1:springer, 2:summer, 3:fall, 4:winter)
	- yr : year (0: 2011, 1:2012)
	- mnth : month ( 1 to 12)
	- hr : hour (0 to 23)
	- holiday : weather day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
	- weekday : day of the week
	- workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
	+ weathersit : 
		- 1: Clear, Few clouds, Partly cloudy, Partly cloudy
		- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
		- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
		- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
	- temp : Normalized temperature in Celsius. The values are divided to 41 (max)
	- atemp: Normalized feeling temperature in Celsius. The values are divided to 50 (max)
	- hum: Normalized humidity. The values are divided to 100 (max)
	- windspeed: Normalized wind speed. The values are divided to 67 (max)
	- casual: count of casual users
	- registered: count of registered users
	- cnt: count of total rental bikes including both casual and registered
    
From the above, we want to use the features season, yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, atemp, hum and windspeed to predict cnt. 


```python
schemaNames = rawdata.schema.names
ncolumns = len(rawdata.columns)
new_rawdata = rawdata.select(schemaNames[2:ncolumns])
```

Transform to DoubleType


```python
new_schemaNames = new_rawdata.schema.names
from pyspark.sql.types import DoubleType
new_ncolumns = len(new_rawdata.columns)
for i in range(new_ncolumns):
    new_rawdata = new_rawdata.withColumn(new_schemaNames[i], new_rawdata[new_schemaNames[i]].cast(DoubleType()))
```


```python
new_rawdata.printSchema()
```

    root
     |-- season: double (nullable = true)
     |-- yr: double (nullable = true)
     |-- mnth: double (nullable = true)
     |-- hr: double (nullable = true)
     |-- holiday: double (nullable = true)
     |-- weekday: double (nullable = true)
     |-- workingday: double (nullable = true)
     |-- weathersit: double (nullable = true)
     |-- temp: double (nullable = true)
     |-- atemp: double (nullable = true)
     |-- hum: double (nullable = true)
     |-- windspeed: double (nullable = true)
     |-- casual: double (nullable = true)
     |-- registered: double (nullable = true)
     |-- cnt: double (nullable = true)
    


We now create the training and test data


```python
(trainingData, testData) = new_rawdata.randomSplit([0.7, 0.3], 42)
```

And assemble the features into a vector


```python
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols = new_schemaNames[0:new_ncolumns-3], outputCol = 'features') 
```

We now want to proceed to apply Poisson Regression over our dataset. We will use the [GeneralizedLinearRegression](https://spark.apache.org/docs/3.0.1/api/python/pyspark.ml.html?highlight=generalizedlinearregression#pyspark.ml.regression.GeneralizedLinearRegression) model for which we can set the following parameters

> **maxIter**: max number of iterations.<p>
    **regParameter**: regularization parameter (>= 0). By setting this parameter to be >0 we are applying an $\ell_2$ regularization.<p>
**familiy**: The name of family which is a description of the error distribution to be used in the model. Supported options: gaussian (default), binomial, poisson, gamma and tweedie.<p>
    **link**: The name of link function which provides the relationship between the linear predictor and the mean of the distribution function. Supported options: identity, log, inverse, logit, probit, cloglog and sqrt. <p>
    The Table below shows the combinations of **family** and **link** functions allowed in this version of PySpark.<p>
        
<table>
<tr><td><b>Family</b></td><td><b>Response Type</b></td><td><b>Supported Links</b></td></tr>
<tr><td>Gaussian</td><td>Continuous</td><td>Identity, Log, Inverse</td></tr>
<tr><td>Binomial</td><td>Binary</td><td>Logit, Probit, CLogLog</td></tr>
<tr><td>Poisson</td><td>Count</td><td>Log, Identity, Sqrt</td></tr>
<tr><td>Gamma</td><td>Continuous</td><td>Inverse, Identity, Log</td></tr>
<tr><td>Tweedie</td><td>Zero-inflated continuous</td><td>Power link function</td></tr>
</table>    


```python
from pyspark.ml.regression import GeneralizedLinearRegression
glm_poisson = GeneralizedLinearRegression(featuresCol='features', labelCol='cnt', maxIter=50, regParam=0.01,\
                                          family='poisson', link='log')
```

We now create a Pipeline


```python
from pyspark.ml import Pipeline
stages = [assembler, glm_poisson]
pipeline = Pipeline(stages=stages)
```

We fit the pipeline to the dataset


```python
pipelineModel = pipeline.fit(trainingData)
```

We now evaluate the RMSE


```python
predictions = pipelineModel.transform(testData)
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator\
      (labelCol="cnt", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("RMSE = %g " % rmse)
```

    RMSE = 142.214 



```python
pipelineModel.stages[-1].coefficients
```




    DenseVector([0.133, 0.4267, 0.002, 0.0477, -0.1086, 0.005, 0.015, -0.0633, 0.7031, 0.6608, -0.746, 0.2307])



## 2. Exercises

**Note**: A *reference* solution will be provided in Blackboard for this part by the following Monday.


### Exercise 1

The variables season, yr, mnth, hr, holiday, weekday, workingday and weathersit are categorical variables that have been treated as continuous variables. In general this is not optimal since we are indirectly imposing a geometry or order over a variable that does not need to have such geometry. For example, the variable season takes values 1 (spring), 2 (summer), 3 (fall) and 4 (winter). Indirectly, we are saying that the distance between spring and winter (1 and 4) is larger than the distance between spring (1) and summer (3). There is not really a reason for this. To avoid this imposed geometries over variables that do not follow one, the usual approach is to transform categorical features to a representation of one-hot encoding. Use the [OneHotEncoder](https://spark.apache.org/docs/3.0.1/api/python/pyspark.ml.html#pyspark.ml.feature.OneHotEncoder) estimator over the Bike Sharing Dataset to represent the categorical variables. Using the same training and test data compute the RMSE over the test data using the same Poisson model. 

### Exercise 2

Compare the performance of Linear Regression over the same dataset using the following algorithms: 

1. Linear Regression using $\ell_1$ regularisation and optimisation OWL-QN.
2. Linear Regression using elasticNet regularisation and optimisation OWL-QN.
3. Linear Regression using $\ell_2$ regularisation and optimisation L-BFGS.
4. Linear Regression using $\ell_2$ regularisation and optimisation IRLS.

## 3. Additional exercise (optional)

**Note**: NO solutions will be provided for this part.

The type of features used for regression can have a dramatic influence over the performance. When we use one-hot encoding for the categorical features, the prediction error of the Poisson regression reduces considerable (see Exercise 1). We could further preprocess the features to see how the preprocessing can influence the performance. Test the performance of Poisson regression and the Linear Regression models in Exercise 2 when the continuous features are standardized (the mean of each feature is made equal to zero and the standard deviation is equal to one). Standardization is performed over the training data only, and the means and standard deviations computed over the training data are later used to standardize the test data. 
