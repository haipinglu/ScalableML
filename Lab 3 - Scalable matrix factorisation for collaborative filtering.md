# Lab 3: Matrix Factorisation for Collaborative Filtering in Recommender Systems

[COM6012 Scalable Machine Learning **2021**](https://github.com/haipinglu/ScalableML) by [Haiping Lu](http://staffwww.dcs.shef.ac.uk/people/H.Lu/) at The University of Sheffield

## Study schedule

- [Task 1](#1-Movie-recommendation-via-collaborative-filtering): To finish by Wednesday. **Essential**
- [Task 2](#2-exercises): To finish by Thursday. ***Exercise***
- [Task 3](#3-additional-ideas-to-explore-optional): To explore further. *Optional*

### Suggested reading

- [Collaborative Filtering in Spark](https://spark.apache.org/docs/3.0.1/ml-collaborative-filtering.html)
- [DataBricks movie recommendations tutorial](https://github.com/databricks/spark-training/blob/master/website/movie-recommendation-with-mllib.md). [DataBricks](https://en.wikipedia.org/wiki/Databricks) is a company founded by the creators of Apache Spark. Check out their packages at [their GitHub page](https://github.com/databricks). They offer a free (up to 15GB memory) cloud computing platform [Databricks Community Edition](https://community.cloud.databricks.com/login.html) that you can try out.
- [Collaborative Filtering on Wiki](http://en.wikipedia.org/wiki/Recommender_system#Collaborative_filtering)
- [Python API on ALS for recommender system](https://spark.apache.org/docs/3.0.1/api/python/pyspark.ml.html#pyspark.ml.recommendation.ALS)
- Chapter *ALS: Stock Portfolio Recommendations* (particularly Section *Demo*) of [PySpark tutorial](https://runawayhorse001.github.io/LearningApacheSpark/pyspark.pdf) 

### PySpark APIs in pictures (highly recommended)

- [**Learn PySpark APIs via Pictures**](https://github.com/jkthompson/pyspark-pictures) (**from recommended repositories** in GitHub, i.e., found via **recommender systems**!)

## 1. Movie recommendation via collaborative filtering

### Get started

Let's start a pyspark shell with 2 cores using a regular queue:

   ```sh
   qrshx -pe smp 2 # request 2 CPU cores without using our reserved queue
   source myspark.sh # myspark.sh should be under the root directory
   cd com6012/ScalableML # our main working directory
   pyspark --master local[2] # start pyspark with 2 cores requested above.
  ```

### Collaborative filtering

[Collaborative filtering](http://en.wikipedia.org/wiki/Recommender_system#Collaborative_filtering) is a classic approach for recommender systems. These techniques aim to fill in the missing entries of a user-item association matrix primarily based on the matrix *itself*.  `spark.ml` currently supports **model-based** collaborative filtering, in which users and products are described by a small set of latent factors that can be used to predict missing entries, using the **alternating least squares (ALS)** algorithm. 

[API](https://spark.apache.org/docs/3.0.1/api/python/pyspark.ml.html#module-pyspark.ml.recommendation): `class pyspark.ml.recommendation.ALS(rank=10, maxIter=10, regParam=0.1, numUserBlocks=10, numItemBlocks=10, implicitPrefs=False, alpha=1.0, userCol='user', itemCol='item', seed=None, ratingCol='rating', nonnegative=False, checkpointInterval=10, intermediateStorageLevel='MEMORY_AND_DISK', finalStorageLevel='MEMORY_AND_DISK', coldStartStrategy='nan', blockSize=4096)`

The following parameters are available:

- *rank*: the number of latent factors in the model (defaults to 10).
- *maxIter* is the maximum number of iterations to run (defaults to 10).
- *regParam*: the regularization parameter in ALS (defaults to 1.0).
- *numUserBlocks*/*numItemBlocks*: the number of blocks the users and items will be partitioned into in order to parallelize computation (defaults to 10).
- *implicitPrefs*: whether to use the explicit feedback ALS variant or one adapted for implicit feedback data (defaults to false which means using explicit feedback).
- *alpha*: a parameter applicable to the implicit feedback variant of ALS that governs the baseline confidence in preference observations (defaults to 1.0).
- *nonnegative*: whether or not to use nonnegative constraints for least squares (defaults to false).
- *coldStartStrategy*: can be set to “drop” in order to drop any rows in the DataFrame of predictions that contain NaN values (defaults to "nan", assigning NaN to a user and/or item factor is not present in the model.
- *blockSize*: the size of the user/product blocks in the blocked implementation of ALS (to reduce communication).

### Movie recommendation

In the cells below, we present a small example of collaborative filtering with the data taken from the [MovieLens](http://grouplens.org/datasets/movielens/) project. Here, we use the old 100k dataset, which has been downloaded in the `Data` folder but you are encouraged to view the source.

The dataset looks like this:

```markdown
    196     242     3       881250949
    186     302     3       891717742
    22      377     1       878887116
    244     51      2       880606923
```

This is a **tab separated** list of 

```markdown
    user id | item id | rating | timestamp 
```

#### Explicit vs. implicit feedback

The data above is typically viewed as a user-item matrix with the ratings as the entries and users and items determine the row and column indices. The ratings are **explicit feedback**. The *Mean Squared Error* of rating prediction can be used to evaluate the recommendation model.

The ratings can also be used differently. We can treat them as numbers representing the strength in observations of user actions, i.e., as **implicit feedback** similar to the number of clicks, or the cumulative duration someone spent viewing a movie. Such numbers are then related to the level of confidence in observed user preferences, rather than explicit ratings given to items. The model then tries to find latent factors that can be used to predict the expected preference of a user for an item.

#### Cold-start problem

The cold-start problem refers to the cases when some users and/or items in the test dataset were not present during training the model. In Spark, these users and items are either assigned `NaN` (not a number, default) or dropped (option `drop`).

#### MovieLens100k

Let's study ALS for movie recommendation on the [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/).

```python
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
```

Read the data in and split words (tab separated):

```python
lines = spark.read.text("Data/MovieLens100k.data").rdd
parts = lines.map(lambda row: row.value.split("\t"))
```

We need to convert the text (`String`) into numbers (`int` or `float`) and then convert RDD to DataFrame:

```python
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),rating=float(p[2]), timestamp=int(p[3])))
ratings = spark.createDataFrame(ratingsRDD).cache()
```

Check data:

```python
ratings.show(5)
# +------+-------+------+---------+
# |userId|movieId|rating|timestamp|
# +------+-------+------+---------+
# |   196|    242|   3.0|881250949|
# |   186|    302|   3.0|891717742|
# |    22|    377|   1.0|878887116|
# |   244|     51|   2.0|880606923|
# |   166|    346|   1.0|886397596|
# +------+-------+------+---------+
# only showing top 5 rows
```

Check data type:

```python
ratings.printSchema()
# root
#  |-- userId: long (nullable = true)
#  |-- movieId: long (nullable = true)
#  |-- rating: double (nullable = true)
#  |-- timestamp: long (nullable = true)
```

Prepare the training/test data with seed `6012`:

```python
myseed=6012
(training, test) = ratings.randomSplit([0.8, 0.2], myseed)
training = training.cache()
test = test.cache()
```

Build the recommendation model using ALS on the training data. Note we set cold start strategy to `drop` to ensure we don't get NaN evaluation metrics:

```python
als = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop")
model = als.fit(training)
# 21/02/19 09:34:30 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
# 21/02/19 09:34:30 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
# 21/02/19 09:34:31 WARN LAPACK: Failed to load implementation from: com.github.fommil.netlib.NativeSystemLAPACK
# 21/02/19 09:34:31 WARN LAPACK: Failed to load implementation from: com.github.fommil.netlib.NativeRefLAPACK
```

The warnings on BLAS and LAPACK are about [optimised numerical processing](https://spark.apache.org/docs/3.0.1/ml-guide.html#dependencies). The warning messages mean that a pure JVM implementation will be used instead of the optimised ones, which need to be [installed separately](https://spark.apache.org/docs/3.0.1/ml-linalg-guide.html). We are not installing them in this module but if you may try on your own machine (not HPC due to access right) if interested.

Evaluate the model by computing the RMSE on the test data:

```python
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))
# Root-mean-square error = 0.9209573052637269
```

Generate top 10 movie recommendations for each user:

```python
userRecs = model.recommendForAllUsers(10)
userRecs.show(5,  False)
# +------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |userId|recommendations                                                                                                                                                                      |
# +------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |471   |[[1643, 6.5805216], [1605, 5.3627353], [309, 5.275607], [613, 5.146943], [114, 5.0345564], [133, 4.8838453], [1159, 4.8557506], [869, 4.833519], [1302, 4.8330326], [776, 4.8192673]]|
# |463   |[[113, 4.39458], [6, 4.17109], [169, 4.030159], [1005, 3.98091], [1252, 3.9767776], [867, 3.9396996], [516, 3.9310303], [1449, 3.919868], [246, 3.908695], [915, 3.9079194]]         |
# |833   |[[1368, 4.767944], [1021, 4.528815], [1597, 4.501705], [340, 4.4845786], [1463, 4.4715357], [1524, 4.451138], [56, 4.3978567], [1128, 4.372282], [634, 4.359913], [646, 4.26006]]    |
# |496   |[[1589, 4.4875054], [114, 4.471943], [320, 4.4689455], [1240, 4.3582573], [119, 4.3407416], [1085, 4.2807164], [613, 4.2485986], [1643, 4.22283], [475, 4.216228], [390, 4.1804137]] |
# |148   |[[50, 4.7727013], [1664, 4.7229643], [1463, 4.719222], [173, 4.6712356], [169, 4.6699014], [168, 4.635963], [115, 4.607701], [478, 4.5947466], [1367, 4.5931487], [172, 4.5518513]]  |
# +------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# only showing top 5 rows
```

Generate top 10 user recommendations for each movie

```python
movieRecs = model.recommendForAllItems(10)
movieRecs.show(5, False)
# +-------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |movieId|recommendations                                                                                                                                                                   |
# +-------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |1580   |[[38, 1.1592693], [589, 1.0558939], [415, 1.053479], [340, 1.0466049], [580, 1.0436047], [688, 1.0368994], [16, 1.01447], [152, 1.0036571], [475, 0.99923307], [264, 0.9971297]]  |
# |471    |[[341, 4.742011], [628, 4.7299995], [907, 4.711858], [810, 4.7082725], [507, 4.694857], [372, 4.644466], [173, 4.643699], [324, 4.6061506], [688, 4.568334], [849, 4.56669]]      |
# |1591   |[[157, 5.147288], [440, 5.0715895], [681, 4.964819], [861, 4.916547], [810, 4.865007], [212, 4.743086], [696, 4.690889], [443, 4.659895], [697, 4.633425], [512, 4.62862]]        |
# |1342   |[[219, 4.0400357], [565, 3.9486341], [765, 3.9319253], [34, 3.8362799], [662, 3.8200698], [803, 3.8044589], [270, 3.732796], [737, 3.7270598], [449, 3.7244747], [558, 3.6913967]]|
# |463    |[[519, 5.0955424], [928, 5.061705], [440, 4.947975], [810, 4.89811], [173, 4.8721247], [174, 4.761526], [52, 4.744709], [819, 4.7408133], [46, 4.72528], [157, 4.7203035]]        |
# +-------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
only showing top 5 rows
```

Generate top 10 movie recommendations for a specified set of users:

```python
users = ratings.select(als.getUserCol()).distinct().limit(3)
userSubsetRecs = model.recommendForUserSubset(users, 10)
users.show()
# +------+
# |userId|
# +------+
# |    26|
# |    29|
# |   474|
# +------+
userSubsetRecs.show(3,False)
# +------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |userId|recommendations                                                                                                                                                                      |
# +------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |26    |[[1449, 4.050502], [884, 3.9505653], [1122, 3.9379153], [1643, 3.93372], [408, 3.9252207], [483, 3.8798523], [318, 3.8776188], [114, 3.8497207], [119, 3.8116813], [127, 3.810518]]  |
# |474   |[[884, 4.9401073], [318, 4.8852944], [1449, 4.8396335], [1122, 4.811585], [408, 4.7511024], [64, 4.7394896], [357, 4.7208233], [1643, 4.7133975], [127, 4.7032666], [1064, 4.697028]]|
# |29    |[[1449, 4.656543], [884, 4.6300607], [963, 4.581265], [272, 4.550117], [408, 4.52602], [114, 4.4871216], [318, 4.4804883], [483, 4.4729433], [64, 4.456217], [1122, 4.4341135]]      |
# +------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

Generate top 10 user recommendations for a specified set of movies

```python
movies = ratings.select(als.getItemCol()).distinct().limit(3)
movieSubSetRecs = model.recommendForItemSubset(movies, 10)
movies.show()
# +-------+
# |movieId|
# +-------+
# |    474|
# |     29|
# |     26|
# +-------+
movieSubSetRecs.show(3,False)
# +-------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |movieId|recommendations                                                                                                                                                                  |
# +-------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |26     |[[270, 4.4144955], [341, 4.3995667], [366, 4.3244834], [770, 4.249518], [118, 4.2322574], [414, 4.20005], [274, 4.184001], [923, 4.1715975], [173, 4.168408], [180, 4.1619034]]  |
# |474    |[[928, 5.138258], [810, 5.0860457], [173, 5.0576525], [239, 5.0229735], [794, 4.9939513], [747, 4.9605308], [310, 4.9474325], [686, 4.904606], [339, 4.896157], [118, 4.8948402]]|
# |29     |[[127, 4.425912], [507, 4.315676], [427, 4.2933187], [811, 4.264675], [472, 4.193143], [628, 4.180931], [534, 4.0719366], [907, 4.0206494], [939, 3.993926], [677, 3.967888]]    |
# +-------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

Let us take a look at the learned factors:

```python
dfItemFactors=model.itemFactors
dfItemFactors.show()
# +---+--------------------+
# | id|            features|
# +---+--------------------+
# | 10|[0.19747244, 0.65...|
# | 20|[0.7627246, 0.401...|
# | 30|[0.6798929, 0.540...|
# | 40|[0.3069924, 0.470...|
# | 50|[-0.14568746, 1.0...|
# | 60|[0.9483578, 1.100...|
# | 70|[0.38704985, 0.29...|
# | 80|[-0.34140116, 1.0...|
# | 90|[0.25251853, 1.20...|
# |100|[0.55230147, 0.68...|
# |110|[0.01884227, 0.20...|
# |120|[0.044651568, 0.1...|
# |130|[-0.22696862, -0....|
# |140|[-0.10749636, 0.1...|
# |150|[0.5518842, 1.098...|
# |160|[0.44457814, 0.80...|
# |170|[0.54556036, 0.77...|
# |180|[0.4252759, 0.709...|
# |190|[0.2943792, 0.781...|
# |200|[0.34967986, 0.44...|
# +---+--------------------+
# only showing top 20 rows
```

`describe().show()` is very handy to inspect your (big) data for understanding/debugging. Try to use it more often to see.

```python
dfItemFactors.describe().show()
# +-------+-----------------+
# |summary|               id|
# +-------+-----------------+
# |  count|             1652|
# |   mean|830.6483050847457|
# | stddev|481.7646649639072|
# |    min|                1|
# |    max|             1682|
# +-------+-----------------+

allmovies = ratings.select(als.getItemCol()).distinct()
allmovies.describe().show()
# +-------+-----------------+
# |summary|          movieId|
# +-------+-----------------+
# |  count|             1682|
# |   mean|            841.5|
# | stddev|485.6958925088827|
# |    min|                1|
# |    max|             1682|
# +-------+-----------------+
```

## 2. Exercises

### More movie recommendations

Do the following on HPC. Run your code in batch mode to produce the results.

1. Download the MovieLens [ml-latest-small](http://files.grouplens.org/datasets/movielens/ml-latest-small.zip) dataset using `wget` as in Lab 2 exercises to the `ScalableML/Data` directory on HPC. Use the `unzip` command to unzip the files to a directory of your choice (search "unzip linux" to see examples of usage). Read the [readme](http://files.grouplens.org/datasets/movielens/ml-latest-small-README.html) for this dataset to understand the data.
2. Use `ALS` to learn five recommendation models on this dataset, using the same split ratio (`0.8, 0.2`) and seed (`6012`) as above but five different values of the `rank` parameter: 5, 10, 15, 20, 25. Plot the (five) resulting RMSE values (on the test set) against the five rank values.
3. Find the top five movies to recommend to any one user of your choice and display the titles and genres for these five movies (via programming).

## 3. Additional ideas to explore (*optional*)

### Databricks tutorial

- Complete the tasks in the [quiz provided by DataBricks](https://github.com/databricks/spark-training/blob/master/machine-learning/python/MovieLensALS.py) on their data or the data from MovieLens directly. [A solution](https://github.com/databricks/spark-training/blob/master/machine-learning/python/solution/MovieLensALS.py) is available but you should try before consulting the solution.

### Santander Kaggle competition on produce recommendation

- A Kaggle competition on [Santander Product Recommendation](https://www.kaggle.com/c/santander-product-recommendation) with a prize of **USD 60,000**, and **1,779 teams** participating.
- Follow this [PySpark notebook on an ALS-based solution](https://www.elenacuoco.com/2016/12/22/alternating-least-squares-als-spark-ml/).
- Learn the way to consider **implicit preferences** and do the same for other recommendation problems.

### Stock Portfolio Recommendations

- Follow Chapter *ALS: Stock Portfolio Recommendations* of [PySpark tutorial](https://runawayhorse001.github.io/LearningApacheSpark/pyspark.pdf)  to perform [Stock Portfolio Recommendations](https://en.wikipedia.org/wiki/Portfolio_investment))
- The data can be downloaded from [Online Retail Data Set](https://archive.ics.uci.edu/ml/datasets/online+retail) at UCI.
- Pay attention to the **data cleaning** step that removes rows containing null value. You may need to do the same when you are dealing with real data.
- The data manipulation steps are useful to learn.

### Context-aware recommendation and time-split evaluation

- See the method in [Joint interaction with context operation for collaborative filtering](https://www.sciencedirect.com/science/article/pii/S0031320318304242?dgcid=rss_sd_all) and implement it in PySpark.
- Perform the **time split recommendation** as discussed in the paper for the above recommender systems.
