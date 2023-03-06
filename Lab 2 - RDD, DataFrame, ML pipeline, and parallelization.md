# Lab 2 - RDD, DataFrame, ML Pipeline, and Parallelization

[COM6012 Scalable Machine Learning **2023**](https://github.com/haipinglu/ScalableML) by [Haiping Lu](https://haipinglu.github.io/) at The University of Sheffield

**Accompanying lectures**: [YouTube video lectures recorded in Year 2020/21.](https://www.youtube.com/playlist?list=PLuRoUKdWifzxj5tm-FZRhs_DeBi2hra-d)

## Study schedule

- [Task 1](#1-rdd-and-shared-variables): To finish in the lab session on 17th Feb. **Essential**
- [Task 2](#2-dataframe): To finish in the lab session on 17th Feb. **Essential**
- [Task 3](#3-machine-learning-library-and-pipelines): To finish in the lab session on 17th Feb. **Essential**
- [Task 4](#4-exercises): To finish by the following Wednesday 22nd Feb. ***Exercise***
- [Task 5](#5-additional-ideas-to-explore-optional): To explore further. *Optional*

### Suggested reading

- Chapters 5 and 6, and especially **Section 9.1** (of Chapter 9)  of [PySpark tutorial](https://runawayhorse001.github.io/LearningApacheSpark/pyspark.pdf)
- [RDD Programming Guide](https://spark.apache.org/docs/3.3.1/rdd-programming-guide.html): Most are useful to know in this module.
- [Spark SQL, DataFrames and Datasets Guide](https://spark.apache.org/docs/3.3.1/sql-programming-guide.html): `Overview` and `Getting Started` recommended (skipping those without Python example).
- [Machine Learning Library (MLlib) Guide](https://spark.apache.org/docs/3.3.1/ml-guide.html)
- [ML Pipelines](https://spark.apache.org/docs/3.3.1/ml-pipeline.html)
- [Apache Spark Examples](https://spark.apache.org/examples.html)
- [Basic Statistics - DataFrame API](https://spark.apache.org/docs/3.3.1/ml-statistics.html)
- [Basic Statistics - RDD API](https://spark.apache.org/docs/3.3.1/mllib-statistics.html): much **richer**

### Compact references (highly recommended)

- [Cheat sheet PySpark Python](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PySpark_Cheat_Sheet_Python.pdf)
- [Cheat sheet PySpark SQL Python](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PySpark_SQL_Cheat_Sheet_Python.pdf)
- [Cheat sheet for PySpark (2 page version)](https://github.com/runawayhorse001/CheatSheet/blob/master/cheatSheet_pyspark.pdf)

**Tip**: Try to use as much DataFrame APIs as possible by referring to the [PySpark API documentation](https://spark.apache.org/docs/3.3.1/api/python/index.html). When you try to program something, try to search whether there is a function in the API already.

## 1. RDD and Shared Variables

### Get started

Firstly, we follow the standard steps as in Task 2 of Lab 1 but with some variations in settings, i.e. to request **4 cores** for an interactive shell. If the request *could not be scheduled*, try to reduce the number of cores requested. We also install `numpy` to our environment for later use.

   ```sh
   qrshx -P rse-com6012 -pe smp 4 # request 4 CPU cores using our reserved queue
   source myspark.sh # assuming HPC/myspark.sh is under your root directory, otherwise, see Lab 1 Task 2
   conda install -y numpy # install numpy, to be used in Task 3. This ONLY needs to be done ONCE. NOT every time.
   cd com6012/ScalableML # our main working directory
   pyspark --master local[4] # start pyspark with 4 cores requested above.
  ```

As stated in the [RDD Programming Guide](https://spark.apache.org/docs/3.3.1/rdd-programming-guide.html#parallelized-collections), Spark allows for parallel operations in a program to be executed on a cluster with the following abstractions:

- **Main abstraction**: **resilient distributed dataset (RDD)** is a collection of elements partitioned across the nodes of the cluster that can be operated on in parallel.
- **Second abstraction**: **shared variables** can be shared across tasks, or between tasks and the driver program. Two types:
  - *Broadcast variables*, which can be used to cache a value in memory on all nodes
  - *Accumulators*, which are variables that are only “added” to, such as counters and sums.

### Parallelized collections (RDDs)

[Parallelized collections](https://spark.apache.org/docs/3.3.1/rdd-programming-guide.html#parallelized-collections) are created by calling `SparkContext`’s `parallelize` method on an existing iterable or collection in your driver program. The elements of the collection are copied to form a distributed dataset that can be operated on in parallel. For example, here is how to create a parallelized collection holding the numbers 1 to 5:

```python
data = [1, 2, 3, 4, 5]
rddData = sc.parallelize(data)
rddData.collect()
# [1, 2, 3, 4, 5] 
```

**Note**: From Lab 2, I will show output as comments `# OUTPUT` and show code without `>>>` for easy copy and paste in the shell.

The number of *partitions* can be set manually by passing parallelize a second argument to the SparkContext

```python
sc.parallelize(data, 16)
# ParallelCollectionRDD[1] at readRDDFromFile at PythonRDD.scala:274
```

Spark tries to set the number of partitions automatically based on the cluster, the rule being *2-4 partitions for every CPU in the cluster*.

### $\pi$ Estimation

Spark can also be used for compute-intensive tasks. This code estimates $\pi$ by "throwing darts" at a circle. We pick random points in the unit square ((0, 0) to (1,1)) and see how many fall in the unit circle. The fraction should be $\pi / 4$ so we use this to get our estimate. This is because the area of a unit circle is $\pi$. See [a visual illustration here](https://docs.ovh.com/gb/en/data-processing/pi-spark/). [`random.random()`](https://docs.python.org/3/library/random.html) returns the next random floating point number in the range [0.0, 1.0).

```python
from random import random

def inside(p):
    x, y = random(), random()
    return x*x + y*y < 1

NUM_SAMPLES = 10000000
count = sc.parallelize(range(0, NUM_SAMPLES),8).filter(inside).count()
print("Pi is roughly %f" % (4.0 * count / NUM_SAMPLES))
# Pi is roughly 3.140986
```

Note that we did not control the seed above so you are not likely to get exactly the same number `3.139747`. You can change `NUM_SAMPLES` to see the difference in precision and time cost.

### [Shared Variables](https://spark.apache.org/docs/3.3.1/rdd-programming-guide.html#shared-variables)

When a function passed to a Spark operation (such as `map` or `reduce`) is executed on a remote cluster node, it works on *separate* copies of all the variables used in the function. These variables are *copied* to each machine, and no updates to the variables on the remote machine are propagated back to the driver program. Supporting general, read-write shared variables across tasks would be inefficient. However, Spark does provide two limited types of shared variables for two common usage patterns: broadcast variables and accumulators.

#### [Broadcast variables](https://spark.apache.org/docs/3.3.1/rdd-programming-guide.html#broadcast-variables)

To avoid creating a copy of a **large** variable for each task, an accessible (*read-only*!) variable can be kept on each machine - this is useful for particularly large datasets which may be needed for multiple tasks. The data broadcasted this way is cached in [serialized](https://spark.apache.org/docs/3.3.1/tuning.html#serialized-rdd-storage) form and deserialized before running each task. See [Data Serialization](https://spark.apache.org/docs/3.3.1/tuning.html#data-serialization) for more details about serialization.

Broadcast variables are created from a variable $v$ by calling `SparkContext.broadcast(v)`. The broadcast variable is a wrapper around $v$, and its value can be accessed by calling the *value* method.

```python
broadcastVar = sc.broadcast([1, 2, 3])
broadcastVar
# <pyspark.broadcast.Broadcast object at 0x2ac15e088860>
broadcastVar.value
# [1, 2, 3]
```

#### [Accumulators](https://spark.apache.org/docs/3.3.1/rdd-programming-guide.html#accumulators)

Accumulators are variables that are only “added” to through an associative and commutative operation and can therefore be efficiently supported in parallel. They can be used to implement counters (as in `MapReduce`) or sums.

An accumulator is created from an initial value `v` by calling `SparkContext.accumulator(v)`
Cluster tasks can then add to it using the `add` method. However, they cannot read its value. Only the dirver program can read the accumulator's value using its `value` method.

```python
accum = sc.accumulator(0)
accum
# Accumulator<id=0, value=0>
sc.parallelize([1, 2, 3, 4]).foreach(lambda x: accum.add(x))
accum.value
# 10
sc.parallelize([1, 2, 3, 4]).foreach(lambda x: accum.add(x))
accum.value
# 20
```

## 2. DataFrame

Along with the introduction of `SparkSession`, the `resilient distributed dataset` (RDD) was replaced by [`dataset`](http://spark.apache.org/docs/3.3.1/api/scala/index.html#org.apache.spark.sql.Dataset). Again, these are objects that can be worked on in parallel. The available operations are:

- **transformations**: produce new datasets
- **actions**: computations which return results

We will start with creating dataframes and datasets, showing how we can print their contents. We create a dataframe in the cell below and print out some info (we can also modify the output before printing):

From RDD to DataFrame

```python
rdd = sc.parallelize([(1,2,3),(4,5,6),(7,8,9)])
df = rdd.toDF(["a","b","c"])
rdd
# ParallelCollectionRDD[10] at readRDDFromFile at PythonRDD.scala:274 
```

The number in `[ ]` (`10`) is the index for RDDs in the shell so it may vary.

Let us examine the DataFrame

```python
df
# DataFrame[a: bigint, b: bigint, c: bigint]
df.show()
# +---+---+---+
# |  a|  b|  c|
# +---+---+---+
# |  1|  2|  3|
# |  4|  5|  6|
# |  7|  8|  9|
# +---+---+---+
df.printSchema()
# root
#  |-- a: long (nullable = true)
#  |-- b: long (nullable = true)
#  |-- c: long (nullable = true)
```

Now let us get RDD from DataFrame

```python
rdd2=df.rdd
rdd2
# MapPartitionsRDD[26] at javaToPython at NativeMethodAccessorImpl.java:0
rdd2.collect()  # view the content
# [Row(a=1, b=2, c=3), Row(a=4, b=5, c=6), Row(a=7, b=8, c=9)]
```

#### Load data from a CSV file

This data was downloaded from a [classic book on statistical learning](https://www.statlearning.com/).

```python
df = spark.read.load("Data/Advertising.csv", format="csv", inferSchema="true", header="true")
df.show(5)  # show the top 5 rows
# 21/02/13 12:10:03 WARN CSVHeaderChecker: CSV header does not conform to the schema.
#  Header: , TV, radio, newspaper, sales
#  Schema: _c0, TV, radio, newspaper, sales
# Expected: _c0 but found:
# CSV file: file:///home/ac1hlu/com6012/ScalableML/Data/Advertising.csv
# +---+-----+-----+---------+-----+
# |_c0|   TV|radio|newspaper|sales|
# +---+-----+-----+---------+-----+
# |  1|230.1| 37.8|     69.2| 22.1|
# |  2| 44.5| 39.3|     45.1| 10.4|
# |  3| 17.2| 45.9|     69.3|  9.3|
# |  4|151.5| 41.3|     58.5| 18.5|
# |  5|180.8| 10.8|     58.4| 12.9|
# +---+-----+-----+---------+-----+
# only showing top 5 rows
```

Note that a warning is given because the first column has an empty header. If we manually specify it, e.g. as `index`, the warning will disappear.

Recall that CSV files are semi-structured data so here Spark inferred the scheme automatically. Let us take a look.

```python
df.printSchema()
# root
#  |-- _c0: integer (nullable = true)
#  |-- TV: double (nullable = true)
#  |-- radio: double (nullable = true)
#  |-- newspaper: double (nullable = true)
#  |-- sales: double (nullable = true)
```

Let us remove the first column

```python
df2=df.drop('_c0')
df2.printSchema()
# root
#  |-- TV: double (nullable = true)
#  |-- radio: double (nullable = true)
#  |-- newspaper: double (nullable = true)
#  |-- sales: double (nullable = true)
```

We can get **summary statistics** for numerical columns using **`.describe().show()`**, very handy to inspect your (big) data for understanding/debugging.

```python
df2.describe().show()
# +-------+-----------------+------------------+------------------+------------------+
# |summary|               TV|             radio|         newspaper|             sales|
# +-------+-----------------+------------------+------------------+------------------+
# |  count|              200|               200|               200|               200|
# |   mean|         147.0425|23.264000000000024|30.553999999999995|14.022500000000003|
# | stddev|85.85423631490805|14.846809176168728| 21.77862083852283| 5.217456565710477|
# |    min|              0.7|               0.0|               0.3|               1.6|
# |    max|            296.4|              49.6|             114.0|              27.0|
# +-------+-----------------+------------------+------------------+------------------+
```

## 3. Machine Learning Library and Pipelines

[MLlib](https://spark.apache.org/docs/3.3.1/ml-guide.html) is Spark’s machine learning (ML) library. It provides:

- *ML Algorithms*: common learning algorithms such as classification, regression, clustering, and collaborative filtering
- *Featurization*: feature extraction, transformation, dimensionality reduction, and selection
- *Pipelines*: tools for constructing, evaluating, and tuning ML Pipelines
- *Persistence*: saving and load algorithms, models, and Pipelines
- *Utilities*: linear algebra, statistics, data handling, etc.

`MLlib` allows easy combination of numerous algorithms into a single pipeline using standardized APIs for machine learning algorithms. The key concepts are:

- **Dataframe**. Dataframes can hold a variety of data types.
- **Transformer**. Transforms one dataframe into another.
- **Estimator**. Algorithm which can be fit on a DataFrame to produce a Transformer.
- **Pipeline**. A Pipeline chains multiple Transformers and Estimators together to specify an ML workflow.
- **Parameter**. Transformers and Estimators share a common API for specifying parameters.

A list of some of the available ML features is available [here](http://spark.apache.org/docs/3.3.1/ml-features.html).

**Clarification on whether Estimator is a transformer**. See [Estimators](https://spark.apache.org/docs/3.3.1/ml-pipeline.html#estimators)
> An Estimator abstracts the concept of a learning algorithm or any algorithm that fits or trains on data. Technically, an Estimator implements a method fit(), which accepts a DataFrame and produces a Model, which is a Transformer. For example, a learning algorithm such as LogisticRegression is an Estimator, and calling fit() trains a LogisticRegressionModel, which is a Model and hence a **Transformer**.

### Example: Linear Regression for Advertising

The example below is based on **Section 9.1** of [PySpark tutorial](https://runawayhorse001.github.io/LearningApacheSpark/pyspark.pdf).

#### Convert the data to dense vector (features and label)

Let us convert the above data in CSV format to a typical (feature, label) pair for supervised learning. Here we use the [`Vectors` API](https://spark.apache.org/docs/3.3.1/api/python/reference/api/pyspark.ml.linalg.Vectors.html). You may also review the [lambda expressions in python](https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions).

```python
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors

def transData(data):
    return data.rdd.map(lambda r: [Vectors.dense(r[:-1]),r[-1]]).toDF(['features','label'])

transformed= transData(df2)
transformed.show(5)
# +-----------------+-----+
# |         features|label|
# +-----------------+-----+
# |[230.1,37.8,69.2]| 22.1|
# | [44.5,39.3,45.1]| 10.4|
# | [17.2,45.9,69.3]|  9.3|
# |[151.5,41.3,58.5]| 18.5|
# |[180.8,10.8,58.4]| 12.9|
# +-----------------+-----+
# only showing top 5 rows
```

The labels here are real numbers and this is a **regression** problem. For **classification** problem, you may need to transform labels (e.g., *disease*,*healthy*) to indices with a featureIndexer in Step 5, **Section 9.1** of [PySpark tutorial](https://runawayhorse001.github.io/LearningApacheSpark/pyspark.pdf).

#### Split the data into training and test sets (40% held out for testing)

```python
(trainingData, testData) = transformed.randomSplit([0.6, 0.4], 6012)
```

We set the `seed=6012` in the above (see the [randomSplit API](https://spark.apache.org/docs/3.3.1/api/python/reference/api/pyspark.sql.DataFrame.randomSplit.html) )Check your train and test data as follows. It is a good practice to *keep tracking your data during prototype phase*.

```python
trainingData.show(5)
# +---------------+-----+
# |       features|label|
# +---------------+-----+
# | [4.1,11.6,5.7]|  3.2|
# | [5.4,29.9,9.4]|  5.3|
# |[7.8,38.9,50.6]|  6.6|
# |[8.7,48.9,75.0]|  7.2|
# |[13.1,0.4,25.6]|  5.3|
# +---------------+-----+
# only showing top 5 rows
 
testData.show(5)
# +----------------+-----+
# |        features|label|
# +----------------+-----+
# |  [0.7,39.6,8.7]|  1.6|
# | [7.3,28.1,41.4]|  5.5|
# |  [8.4,27.2,2.1]|  5.7|
# |   [8.6,2.1,1.0]|  4.8|
# |[11.7,36.9,45.2]|  7.3|
# +----------------+-----+
# only showing top 5 rows
```

#### Fit a linear regression Model and perform prediction

More details on parameters can be found in the [Python API documentation](https://spark.apache.org/docs/3.3.1/api/python/reference/api/pyspark.ml.regression.LinearRegression.html).

```python
from pyspark.ml.regression import LinearRegression

lr = LinearRegression()
lrModel = lr.fit(trainingData)
predictions = lrModel.transform(testData)
predictions.show(5)
# +----------------+-----+------------------+
# |        features|label|        prediction|
# +----------------+-----+------------------+
# |  [0.7,39.6,8.7]|  1.6|10.497359087823323|
# | [7.3,28.1,41.4]|  5.5| 8.615626828376815|
# |  [8.4,27.2,2.1]|  5.7|  8.59859112486577|
# |   [8.6,2.1,1.0]|  4.8| 4.027845382391438|
# |[11.7,36.9,45.2]|  7.3| 10.41211129446484|
# +----------------+-----+------------------+
# only showing top 5 rows
```

You may see some warnings, which are normal.

#### Evaluation

```python
from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(labelCol="label",predictionCol="prediction",metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
# Root Mean Squared Error (RMSE) on test data = 1.87125
```

### Example: Machine Learning Pipeline for Document Classification

This example is adapted from the [ML Pipeline API](https://spark.apache.org/docs/3.3.1/ml-pipeline.html), with minor changes and additional explanations.

```python
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
```

Directly create DataFrame (for illustration)

```python
training = spark.createDataFrame([
    (0, "a b c d e spark 6012", 1.0),
    (1, "b d", 0.0),
    (2, "spark f g h 6012", 1.0),
    (3, "hadoop mapreduce", 0.0)
], ["id", "text", "label"])

training.printSchema()
# root
#  |-- id: long (nullable = true)
#  |-- text: string (nullable = true)
#  |-- label: double (nullable = true)

training.show()
# +---+--------------------+-----+
# | id|                text|label|
# +---+--------------------+-----+
# |  0|a b c d e spark 6012|  1.0|
# |  1|                 b d|  0.0|
# |  2|    spark f g h 6012|  1.0|
# |  3|    hadoop mapreduce|  0.0|
# +---+--------------------+-----+
```

Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.

```python
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.001)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
```

Model fitting

```python
model = pipeline.fit(training)
```

Construct test documents (data), which are unlabeled (id, text) tuples

```python
test = spark.createDataFrame([
    (4, "spark i j 6012"),
    (5, "l m n"),
    (6, "spark 6012 spark"),
    (7, "apache hadoop")
], ["id", "text"])

test.show()
# +---+------------------+
# | id|              text|
# +---+------------------+
# |  4|    spark i j 6012|
# |  5|             l m n|
# |  6|spark 6012 spark|
# |  7|     apache hadoop|
# +---+------------------+
```

Make predictions on test documents and print columns of interest.

```python
prediction = model.transform(test)
prediction.show()
# +---+----------------+--------------------+--------------------+--------------------+--------------------+----------+
# | id|            text|               words|            features|       rawPrediction|         probability|prediction|
# +---+----------------+--------------------+--------------------+--------------------+--------------------+----------+
# |  4|  spark i j 6012| [spark, i, j, 6012]|(262144,[19036,11...|[-1.0173918675250...|[0.26553574436761...|       1.0|
# |  5|           l m n|           [l, m, n]|(262144,[1303,526...|[4.76763852580441...|[0.99157121824630...|       0.0|
# |  6|spark 6012 spark|[spark, 6012, spark]|(262144,[111139,1...|[-3.9099070641898...|[0.01964856004327...|       1.0|
# |  7|   apache hadoop|    [apache, hadoop]|(262144,[68303,19...|[5.80789088699039...|[0.99700523688305...|       0.0|
# +---+----------------+--------------------+--------------------+--------------------+--------------------+----------+


selected = prediction.select("id", "text", "probability", "prediction")
selected.show()
# +---+----------------+--------------------+----------+
# | id|            text|         probability|prediction|
# +---+----------------+--------------------+----------+
# |  4|  spark i j 6012|[0.26553574436761...|       1.0|
# |  5|           l m n|[0.99157121824630...|       0.0|
# |  6|spark 6012 spark|[0.01964856004327...|       1.0|
# |  7|   apache hadoop|[0.99700523688305...|       0.0|
# +---+----------------+--------------------+----------+


for row in selected.collect():
    rid, text, prob, prediction = row
    print("(%d, %s) --> prob=%s, prediction=%f" % (rid, text, str(prob), prediction))
# (4, spark i j 6012) --> prob=[0.2655357443676159,0.7344642556323842], prediction=1.000000
# (5, l m n) --> prob=[0.9915712182463081,0.008428781753691883], prediction=0.000000
# (6, spark 6012 spark) --> prob=[0.019648560043272496,0.9803514399567275], prediction=1.000000
# (7, apache hadoop) --> prob=[0.9970052368830581,0.002994763116941912], prediction=0.000000
```

## 4. Exercises

**Note**: A *reference* solution will be provided in Blackboard for this part by the following Thursday.

Starting from this lab, you need to use *as many DataFrame functions as possible*.

### Log mining

1. On HPC, download the description of the NASA access log data to the `Data` directory via

    ```sh
    wget ftp://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html
    ```

    Load the Aug95 NASA access log data in Lab 1 and create a DataFrame with FIVE columns by **specifying** the schema according to the description in the downloaded html file. Use this DataFrame for the following questions.

2. Find out the number of **unique** hosts in total (i.e. in August 1995)?
3. Find out the most frequent visitor, i.e. the host with the largest number of visits.

### Linear regression for advertising

4. Add regularisation to the [linear regression for advertising example](#Example-Linear-Regression-for-Advertising) and evaluate the prediction performance against the performance without any regularisation. Study at least three different regularisation settings.

### Logistic regression for document classification

5. Construct another test dataset for the [machine learning pipeline for document classification example](#Example-Machine-Learning-Pipeline-for-Document-Classification) with three test document samples: `"pyspark hadoop"`; `"spark a b c"`; `"mapreduce spark"` and report the prediction probabilities and the predicted labels for these three sample.

## 5. Additional ideas to explore (*optional*)

**Note**: NO solutions will be provided for this part.

### $\pi$ estimation

- Change the number of partitions to a range of values (e.g. 2, 4, 8, 16, ...) and study the time cost for each value (e.g. by plotting the time cost against the number of partitions).
- Change the number of samples to study the variation in precision and time cost.

### More log mining and machine learning

- Find out the mean and standard deviation of the reply byte size.
- Other questions in Lab 1 Task 6.
- Explore more CSV data of your interest via Google or at [Sample CSV data](https://support.spatialkey.com/spatialkey-sample-csv-data/), including insurance, real estate, and sales transactions.
- Explore the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php) to build machine learning pipelines in PySpark for some datasets of your interest.
