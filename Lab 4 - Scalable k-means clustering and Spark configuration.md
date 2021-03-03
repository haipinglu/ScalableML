# Lab 4: $k$-means clustering and Spark configuration

[COM6012 Scalable Machine Learning **2021**](https://github.com/haipinglu/ScalableML) by [Haiping Lu](http://staffwww.dcs.shef.ac.uk/people/H.Lu/) at The University of Sheffield

## Study schedule

- [Task 1](#1-Space-management-and-Spark-configuration): To finish by Wednesday. **Critical**
- [Task 2](#2-k-means-clustering): To finish by Thursday. **Essential**
- [Task 3](#3-exercises): To finish before the next Monday. ***Exercise***
- [Task 4](#4-additional-ideas-to-explore-optional): To explore further. *Optional*

### Suggested reading

- Chapters *Clustering* and *RFM Analysis* of [PySpark tutorial](https://runawayhorse001.github.io/LearningApacheSpark/pyspark.pdf) 
- [Clustering in Spark](https://spark.apache.org/docs/3.0.1/ml-clustering.html)
- [PySpark API on clustering](https://spark.apache.org/docs/3.0.1/api/python/pyspark.ml.html#pyspark.ml.clustering.KMeans)
- [PySpark code on clustering](https://github.com/apache/spark/blob/master/python/pyspark/ml/clustering.py)
- [$k$-means clustering on Wiki](https://en.wikipedia.org/wiki/K-means_clustering)
- [$k$-means++ on Wiki](https://en.wikipedia.org/wiki/K-means%2B%2B) 
- [$k$-means|| paper](http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf)

## 1. Space management and Spark configuration

As you progress to deal with bigger data, you may need to ensure you have enough disk space and also configure Spark properly to ensure enough resources are available for processing your big data. This section looks at various aspects of space management and Spark configuration.

### Space Management

##### Move to `/data` from `/home`

As you will deal with bigger data, we recommend you to use `/home/YOUR_USERNAME` (10G quota) for setting up `conda` environment (as we did already in Lab 1), and use `/data/YOUR_USERNAME` (100G) for COM6012 works. After logging in, do `cd /data/YOUR_USERNAME` to work under this directory. Let us now move all files for our module to there.

This time, let's request for 4 cores using a regular queue and 10GB of memory, following the [qrshx doc](https://docs.hpc.shef.ac.uk/en/latest/hpc/scheduler/qrshx.html#qrshx). We activate the environment as usual and then install `matplotlib`.

   ```sh
   qrshx -l rmem=10G -pe openmp 4 # request 10GB memory and exit4 CPU cores
   source myspark.sh # myspark.sh should be under the root directory
   conda install -y matplotlib
  ```

If the request *could not be scheduled*, try to reduce the number of cores (and/or the amount of memory) requested and/or use the reserved queue `-P rse-com6012`.

You can either clone `ScalableML` from GitHub to `/data/abc1de/ScalableML` (again `abc1de` should be your username) via

```sh
cd /data/abc1de/
git clone --depth 1 https://github.com/haipinglu/ScalableML
cd ScalableML
```

Or you can copy the whole `ScalableML` to under `data` via

```sh
cp -r /home/abc1de/com6012/ScalableML /data/abc1de/ScalableML
cd /data/abc1de/ScalableML
```

You can check your disk space via

```sh
df -h /home/abc1de
df -h /data/abc1de
```

Start pyspark working under `/data/abc1de/ScalableML` now.

```sh
pyspark --master local[4]
```

### Spark configuration

Take a look at the configuration of the Spark application properties [here (the table)](https://spark.apache.org/docs/3.0.1/configuration.html). There are also several good references: [set spark context;](https://stackoverflow.com/questions/30560241/is-it-possible-to-get-the-current-spark-context-settings-in-pyspark); [set driver memory](https://stackoverflow.com/questions/53606756/how-to-set-spark-driver-memory-in-client-mode-pyspark-version-2-3-1); [set local dir](https://stackoverflow.com/questions/40372655/how-to-set-spark-local-dir-property-from-spark-shell); [a discussion at databricks](https://forums.databricks.com/questions/11856/spark-job-fails-storagediskblockobjectwriter-uncau.html)

Recall that in the provided [`Code/LogMiningBig.py`](Code/LogMiningBig.py), you were asked to set the `spark.local.dir` to `/fastdata/YOUR_USERNAME` in the following

```python
spark = SparkSession.builder \
    .master("local[2]") \
    .appName("COM6012 Spark Intro") \
    .config("spark.local.dir","/fastdata/YOUR_USERNAME") \
    .getOrCreate()
```

In the above, we have configured Spark's `spark.local.dir` property to `/fastdata/YOUR_USERNAME` to use it as a "scratch" space (storing temporal files etc.).

In shell, we can check *customised* (defaults are not shown) config via `sc`:

```python
sc._conf.getAll()
# [('spark.driver.port', '43351'), ('spark.driver.host', 'sharc-node062.shef.ac.uk'), ('spark.master', 'local[4]'), ('spark.sql.catalogImplementation', 'hive'), ('spark.rdd.compress', 'True'), ('spark.serializer.objectStreamReset', '100'), ('spark.submit.pyFiles', ''), ('spark.executor.id', 'driver'), ('spark.submit.deployMode', 'client'), ('spark.app.id', 'local-1614415017491'), ('spark.app.name', 'PySparkShell'), ('spark.ui.showConsoleProgress', 'true')]
```

#### Driver memory and potential `out of memory` problem

The default driver memory (in the configuration table above), i.e., `spark.driver.memory`, is ONLY **1G** so even if you have requested more memory, there can be out of memory problem due to this setting (read the setting description for `spark.driver.memory`). This is true for other memory-related settings as well.

This setting can be changed by setting the configuration option, e.g.,

```sh
spark-submit --driver-memory 8g AS1Q2.py
```

The amount of memory specified about should not exceed the amount you requested when you request a node via `qrshx`. In the beginning above, we requested 10G so if you want 16G, you need to request more memory when you request a node, e.g., starting a session via `qrshx -l rmem=16G` or `qrshx -l rmem=24G`. We can also set this in shell (see references above) but some said that it is too late to change the memory after JVM starts (i.e. at runtime) and **for big data** (even if you see it shows new settings), it is safer to use **spark-submit** to specify such options as above this paragraph.

#### To change more configurations

You may search for example usage, an example that we used in the past **for very big data** is here for your reference only:

```sh
spark-submit --driver-memory 40g --executor-memory 2g --master local[10] -- local.dir /fastdata/USERNAME --conf spark.driver.maxResultSize=4g test.py
```

A reminder that the more resources you request, the longer you need to wait for them to become available to you.

## 2. $k$-means clustering

[$k$-means](http://en.wikipedia.org/wiki/K-means_clustering) is one of the most commonly used clustering algorithms that clusters the data points into a predefined number of clusters. The Spark MLlib implementation includes a parallelized variant of the [$k$-means++](https://en.wikipedia.org/wiki/K-means%2B%2B) method called [$k$-means||](http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf).

`KMeans` is implemented as an `Estimator` and generates a [`KMeansModel`](https://spark.apache.org/docs/3.0.1/api/python/pyspark.ml.html#pyspark.ml.clustering.KMeansModel) as the base model.

[API](https://spark.apache.org/docs/3.0.1/api/python/pyspark.ml.html#pyspark.ml.clustering.KMeans): `class pyspark.ml.clustering.KMeans(featuresCol='features', predictionCol='prediction', k=2, initMode='k-means||', initSteps=2, tol=0.0001, maxIter=20, seed=None, distanceMeasure='euclidean', weightCol=None)`

The following parameters are available:

- *k*: the number of desired clusters.
- *maxIter*: the maximum number of iterations
- *initMode*: specifies either random initialization or initialization via k-means||
- *initSteps*: determines the number of steps in the k-means|| algorithm (default=2, advanced)
- *tol*: determines the distance threshold within which we consider k-means to have converged.
- *seed*: setting the **random seed** (so that multiple runs have the same results)
- *distanceMeasure*: either Euclidean (default) or cosine distance measure
- *weightCol*: optional weighting of data points

We will do some plotting in this lab. We have installed `matplotlib` above. To plot and save figures on HPC, we need to do the following before using pyplot:

```python
import matplotlib 
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab! 
```

Now import modules needed in this lab:

```python
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import Vectors
import matplotlib.pyplot as plt
```

### Clustering of simple synthetic data

Here, we study $k$-means clustering on a simple example with four well-separated data points as the following.

```python
data = [(Vectors.dense([0.0, 0.0]),), (Vectors.dense([1.0, 1.0]),),
        (Vectors.dense([9.0, 8.0]),), (Vectors.dense([8.0, 9.0]),)]
df = spark.createDataFrame(data, ["features"])
kmeans = KMeans(k=2, seed=1)  # Two clusters with seed = 1
model = kmeans.fit(df)
# 21/02/27 21:11:05 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
# 21/02/27 21:11:05 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
```

We examine the cluster centers (centroids) and use the trained model to "predict" the cluster index for a data point.

```python
centers = model.clusterCenters()
len(centers)
# 2
for center in centers:
    print(center)
# [0.5 0.5]
# [8.5 8.5]
model.predict(df.head().features)
# 0
```

We can use the trained model to cluster any data points in the same space, where the cluster index is considered as the `prediction`.

```python
transformed = model.transform(df)
transformed.show()
# +---------+----------+
# | features|prediction|
# +---------+----------+
# |[0.0,0.0]|         0|
# |[1.0,1.0]|         0|
# |[9.0,8.0]|         1|
# |[8.0,9.0]|         1|
# +---------+----------+
```

We can examine the training summary for the trained model.

```python
model.hasSummary
# True
summary = model.summary
 summary
# <pyspark.ml.clustering.KMeansSummary object at 0x2b1662948d30>
summary.k
# 2
summary.clusterSizes
# [2, 2]]
summary.trainingCost  #sum of squared distances of points to their nearest center
# 2.0
```

You can check out the [KMeansSummary API](https://spark.apache.org/docs/3.0.1/api/java/org/apache/spark/ml/clustering/KMeansSummary.html) for details of the summary information, e.g., we can find out that the training cost is the sum of squared distances to the nearest centroid for all points in the training dataset.

### Save and load an algorithm/model

We can save an algorithm/model in a temporary location (see [API on save](https://spark.apache.org/docs/3.0.1/api/python/pyspark.ml.html?highlight=save#pyspark.ml.PipelineModel.save)) and then load it later.

Save and load the $k$-means algorithm (settings):

```python
import tempfile

temp_path = tempfile.mkdtemp()
kmeans_path = temp_path + "/kmeans"
kmeans.save(kmeans_path)
kmeans2 = KMeans.load(kmeans_path)
kmeans2.getK()
# 2
```

Save and load the learned $k$-means model (note that only the learned model is saved, not including the summary):

```python
model_path = temp_path + "/kmeans_model"
model.save(model_path)
model2 = KMeansModel.load(model_path)
model2.hasSummary
# False
model2.clusterCenters()
# [array([0.5, 0.5]), array([8.5, 8.5])]
```

### Iris clustering

Clustering of the [Iris flower data set](https://en.wikipedia.org/wiki/Iris_flower_data_set) is a classical example [discussed the Wikipedia page of $k$-means clustering](https://en.wikipedia.org/wiki/K-means_clustering#Discussion). This data set was introduced by [Ronald Fisher](https://en.wikipedia.org/wiki/Ronald_Fisher), "the father of modern statistics and experimental design" (and thus machine learning) and also "the greatest biologist since Darwin". The code below is based on Chapter *Clustering* of [PySpark tutorial](https://runawayhorse001.github.io/LearningApacheSpark/pyspark.pdf), with some changes introduced.

#### Load and inspect the data

```python
df = spark.read.load("Data/iris.csv", format="csv", inferSchema="true", header="true").cache()
df.show(5,True)
# +------------+-----------+------------+-----------+-------+
# |sepal_length|sepal_width|petal_length|petal_width|species|
# +------------+-----------+------------+-----------+-------+
# |         5.1|        3.5|         1.4|        0.2| setosa|
# |         4.9|        3.0|         1.4|        0.2| setosa|
# |         4.7|        3.2|         1.3|        0.2| setosa|
# |         4.6|        3.1|         1.5|        0.2| setosa|
# |         5.0|        3.6|         1.4|        0.2| setosa|
# +------------+-----------+------------+-----------+-------+
# only showing top 5 rows
df.printSchema()
# root
#  |-- sepal_length: double (nullable = true)
#  |-- sepal_width: double (nullable = true)
#  |-- petal_length: double (nullable = true)
#  |-- petal_width: double (nullable = true)
#  |-- species: string (nullable = true)
```

We can use `.describe().show()` to inspect the (statistics of) data:

```python
df.describe().show()
# +-------+------------------+-------------------+------------------+------------------+---------+
# |summary|      sepal_length|        sepal_width|      petal_length|       petal_width|  species|
# +-------+------------------+-------------------+------------------+------------------+---------+
# |  count|               150|                150|               150|               150|      150|
# |   mean| 5.843333333333335| 3.0540000000000007|3.7586666666666693|1.1986666666666672|     null|
# | stddev|0.8280661279778637|0.43359431136217375| 1.764420419952262|0.7631607417008414|     null|
# |    min|               4.3|                2.0|               1.0|               0.1|   setosa|
# |    max|               7.9|                4.4|               6.9|               2.5|virginica|
# +-------+------------------+-------------------+------------------+------------------+---------+
```

#### Convert the data to dense vector (features)

Use a `transData` function similar to that in Lab 2 to convert the attributes into feature vectors.

```python
def transData(data):
    return data.rdd.map(lambda r: [Vectors.dense(r[:-1])]).toDF(['features'])

dfFeatureVec= transData(df).cache()
dfFeatureVec.show(5, False)
# +-----------------+
# |features         |
# +-----------------+
# |[5.1,3.5,1.4,0.2]|
# |[4.9,3.0,1.4,0.2]|
# |[4.7,3.2,1.3,0.2]|
# |[4.6,3.1,1.5,0.2]|
# |[5.0,3.6,1.4,0.2]|
# +-----------------+
# only showing top 5 rows
```

#### Determine $k$ via silhouette analysis

We can perform a [Silhouette Analysis](https://en.wikipedia.org/wiki/Silhouette_(clustering)) to determine $k$ by running multiple $k$-means with different $k$ and evaluate the clustering results. See [the ClusteringEvaluator API](https://spark.apache.org/docs/3.0.1/api/python/pyspark.ml.html#pyspark.ml.evaluation.ClusteringEvaluator), where `silhouette` is the default metric. You can also refer to this [scikit-learn notebook on the same topic](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html). Other ways of determining the best $k$ can be found on [a dedicated wiki page](https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set).

```python
import numpy as np

numK=10
silhouettes = np.zeros(numK)
costs= np.zeros(numK)
for k in range(2,numK):  # k = 2:9
    kmeans = KMeans().setK(k).setSeed(11)
    model = kmeans.fit(dfFeatureVec)    
    predictions = model.transform(dfFeatureVec)
    costs[k]=model.summary.trainingCost
    evaluator = ClusteringEvaluator()  # to compute the silhouette score
    silhouettes[k] = evaluator.evaluate(predictions)
```

We can take a look at the clustering results (the `prediction` below is the cluster index/label).

```python
predictions.show(15)
# +-----------------+----------+
# |         features|prediction|
# +-----------------+----------+
# |[5.1,3.5,1.4,0.2]|         1|
# |[4.9,3.0,1.4,0.2]|         1|
# |[4.7,3.2,1.3,0.2]|         1|
# |[4.6,3.1,1.5,0.2]|         1|
# |[5.0,3.6,1.4,0.2]|         1|
# |[5.4,3.9,1.7,0.4]|         5|
# |[4.6,3.4,1.4,0.3]|         1|
# |[5.0,3.4,1.5,0.2]|         1|
# |[4.4,2.9,1.4,0.2]|         1|
# |[4.9,3.1,1.5,0.1]|         1|
# |[5.4,3.7,1.5,0.2]|         5|
# |[4.8,3.4,1.6,0.2]|         1|
# |[4.8,3.0,1.4,0.1]|         1|
# |[4.3,3.0,1.1,0.1]|         1|
# |[5.8,4.0,1.2,0.2]|         5|
# +-----------------+----------+
# only showing top 15 rows
```

Plot the cost (sum of squared distances of points to their nearest centroid, the smaller the better) against $k$.

```python
fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,numK),costs[2:numK],marker="o")
ax.set_xlabel('$k$')
ax.set_ylabel('Cost')
plt.grid()
plt.savefig("Output/Lab4_cost.png")
```

We can see that this cost measure is biased towards a large $k$. Let us plot the silhouette metric (the larger the better) against $k$.

```python
fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,numK),silhouettes[2:numK],marker="o")
ax.set_xlabel('$k$')
ax.set_ylabel('Silhouette')
plt.grid()
plt.savefig("Output/Lab4_silhouette.png")
```

We can see that the silhouette measure is biased towards a small $k$. By the silhouette metric, we should choose $k=2$ but we know the ground truth $k$ is 3 (read the [data description](https://archive.ics.uci.edu/ml/datasets/iris) or count unique species). Therefore, this metric is not giving the ideal results in this case (either). [Determining the optimal number of clusters](https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set) is an open problem.

## 3. Exercises

### Further study on iris clustering

Carry out some further studies on the iris clustering problem above.

1. Choose $k=3$ and evaluate the clustering results against the ground truth (class labels) using the [Normalized Mutual Information (NMI) available in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html). You need to install `scikit-learn` in the `myspark` environment via `conda install -y scikit-learn`. This allows us to study the clustering quality when we know the true number of clusters.
2. Use multiple (e.g., 10 or 20) random seeds to generate different clustering results and plot the respective NMI values (with respect to ground truth with $k=3$ as in the question above) to observe the effect of initialisation.

## 4. Additional ideas to explore (*optional*)

### RFM Customer Value Analysis

- Follow Chapter *RFM Analysis* of [PySpark tutorial](https://runawayhorse001.github.io/LearningApacheSpark/pyspark.pdf) to perform [RFM Customer Value Analysis](https://en.wikipedia.org/wiki/RFM_(customer_value))
- The data can be downloaded from [Online Retail Data Set](https://archive.ics.uci.edu/ml/datasets/online+retail) at UCI.
- Note the **data cleaning** step that checks and removes rows containing null value via `.dropna()`. You may need to do the same when you are dealing with real data.
- The **data manipulation** steps are also useful to learn.

### Network intrusion detection

- The original task is a classification task. We can ignore the class labels and perform clustering on the data.
- Write a standalone program (and submit as a batch job to HPC) to do $k$-means clustering on the [KDDCUP1999 data](https://archive.ics.uci.edu/ml/datasets/KDD+Cup+1999+Data) with 4M points. You may start with the smaller 10% subset.

### Color Quantization using K-Means

- Follow the scikit-learn example [Color Quantization using K-Means](https://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html#sphx-glr-auto-examples-cluster-plot-color-quantization-py) to perform the same using PySpark on your high-resolution photos.
