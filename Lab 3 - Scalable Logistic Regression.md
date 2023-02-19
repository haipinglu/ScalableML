# Lab 4: Scalable logistic regression

## Study schedule

- [Section 1](#1-data-storage-in-sharc-and-spark-configuration): To finish by 24th February. **Essential**
- [Section 2](#2-logistic-regression-in-pyspark): To finish by 24th February. **Essential**
- [Section 3](#3-exercises): To finish by the following Thursday 2nd March. ***Exercise***
- [Section 4](#4-additional-exercises-optional): To explore further. *Optional*

## Introduction

In this lab, we will explore the performance of Logistic Regression on the datasets we already used in the Notebook for Decision Trees for Classification, [Lab 3](https://github.com/haipinglu/ScalableML/blob/master/Lab%203%20-%20Scalable%20Decision%20trees.md).

Before we work on Logistic Regression, though, let us briefly look at the different filestore systems available in ShARC and different Spark configurations that are necessary to develop a well performing Spark job.

## 1. Data storage in ShARC and Spark configuration

As you progress to deal with bigger data, you may need to ensure you have enough disk space and also configure Spark properly to ensure enough resources are available for processing your big data. This section looks at various aspects of data storage management and Spark configuration.

### 1.1 Data storage management

#### Move to `/data` from `/home`

As you starting working with larger datasets, we recommend that you use `/home/YOUR_USERNAME` (10G quota) for setting up `conda` environment (as we did already in Lab 1), and that you use `/data/YOUR_USERNAME` (100G) for working on tasks related to COM6012.

After logging into ShARC, do `cd /data/YOUR_USERNAME` to work under this directory. Let us now move all files for our module to there.

This time, let's request four cores using a regular queue and 10GB of memory, following the [qrshx doc](https://docs.hpc.shef.ac.uk/en/latest/referenceinfo/scheduler/SGE/Common-commands/qrshx.html). We activate the environment as usual.

   ```sh
   qrshx -l rmem=10G -pe smp 4 # request 10GB memory and exit4 CPU cores
   source myspark.sh # myspark.sh should be under the root directory
  ```

If the request *could not be scheduled*, try reducing the number of cores (and/or the amount of memory) requested and/or use the reserved queue `-P rse-com6012`.

You can either clone `ScalableML` from GitHub to `/data/abc1de/ScalableML` (again `abc1de` should be your username) via

```sh
cd /data/abc1de/
git clone --depth 1 https://github.com/haipinglu/ScalableML
cd ScalableML
```

Or you can copy the whole `ScalableML` from `home` to `data` using

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

Detailed information about the different storage systems can be found in [this link](https://docs.hpc.shef.ac.uk/en/latest/hpc/filestore.html).

### 1.2 Spark configuration

Take a look at the configuration of the Spark application properties [here (the table)](https://spark.apache.org/docs/latest/configuration.html#application-properties). There are also several good references: [set spark context](https://stackoverflow.com/questions/30560241/is-it-possible-to-get-the-current-spark-context-settings-in-pyspark); [set driver memory](https://stackoverflow.com/questions/53606756/how-to-set-spark-driver-memory-in-client-mode-pyspark-version-2-3-1); [set local dir](https://stackoverflow.com/questions/40372655/how-to-set-spark-local-dir-property-from-spark-shell).

Recall that in the provided [`Code/LogMiningBig.py`](Code/LogMiningBig.py), you were asked to set the `spark.local.dir` to `/fastdata/YOUR_USERNAME` as in the following set of instructions

```python
spark = SparkSession.builder \
    .master("local[2]") \
    .appName("COM6012 Spark Intro") \
    .config("spark.local.dir","/fastdata/YOUR_USERNAME") \
    .getOrCreate()
```

In the instructions above, we have configured Spark's `spark.local.dir` property to `/fastdata/YOUR_USERNAME` to use it as a "scratch" space (e.g. storing temporal files).

In shell, we can check *customised* (defaults are not shown) config via `sc`:

```python
sc._conf.getAll()
# [('spark.driver.port', '43351'), ('spark.driver.host', 'sharc-node062.shef.ac.uk'), ('spark.master', 'local[4]'), ('spark.sql.catalogImplementation', 'hive'), ('spark.rdd.compress', 'True'), ('spark.serializer.objectStreamReset', '100'), ('spark.submit.pyFiles', ''), ('spark.executor.id', 'driver'), ('spark.submit.deployMode', 'client'), ('spark.app.id', 'local-1614415017491'), ('spark.app.name', 'PySparkShell'), ('spark.ui.showConsoleProgress', 'true')]
```

### Driver memory and potential `out of memory` problem

**Pay attention to the memory requirements that you set in the .sh file, and in the spark-submit instructions**

Memory requirements that you request from ShARC are configured in the following two lines appearing in your .sh file

```sh
#!/bin/bash
#$ -pe smp 2 # The smp parallel environment provides multiple cores on one node. <nn> specifies the max number of cores.
#$ -l rmem=8G # -l rmem=xxG is used to specify the maximum amount (xx) of real memory to be requested per CPU core.
```

With the configuration above in the .sh file, we are requesting ShARC for 16GB (2 nodes times 8GB per node) of real memory. If we are working in the `rse-com6012` queue, we are requesting access to one of the five [big memory nodes](https://docs.hpc.shef.ac.uk/en/latest/sharc/groupnodes/big_mem_nodes.html) that we have for this course. We can check we have been allocated to one of these nodes because they are named as `node173` to `node177` in the Linux terminal. Each of these nodes has a total of 768 GB memory and 40 nodes, i.e. 19.2 GB per node. When configuring your .sh file, you need to be careful about how you set these two parameters. In the past, we have seen .sh files intended to be run in one of our nodes with the following configuration

```sh
#!/bin/bash
#$ -pe smp 10 
#$ -l rmem=100G 
```

**Do you see a problem with this configuration?** In this .sh file, they were requesting 1000 GB of memory (10 nodes times 100 GB per node) which exceeds the available memory in each of these nodes, 768 GB.

As well as paying attention to your .sh file for memory requirements, we also need to configure memory requirements in the instructions when we use `spark-submit`, particularly, for the memory that will be allocated to the driver and to each of the executors. The default driver memory, i.e., `spark.driver.memory`, is ONLY **1G** (see [this Table](https://spark.apache.org/docs/3.2.1/configuration.html#application-properties)) so even if you have requested more memory, there can be out of memory problems due to this setting (read the setting description for `spark.driver.memory`). This is true for other memory-related settings as well like the `spark.executor.memory`.

The `spark.driver.memory` option can be changed by setting the configuration option, e.g.,

```sh
spark-submit --driver-memory 8g AS1Q2.py
```

**The amount of memory specified in `driver-memory` above should not exceed the amount you requested to ShARC via your .sh file or qrshx if you are working on interactive mode.**

In the past, we have seen .sh files intended to be run in one of our `rse-com6012` nodes with the following configuration

```sh
#!/bin/bash
#$ -l h_rt=6:00:00  
#$ -pe smp 2 
#$ -l rmem=8G 
#$ -o ../Output/Output.txt  
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M youremail@shef.ac.uk 
#$ -m ea #Email you when it finished or aborted
#$ -cwd 

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit --driver-memory 10g ../Code/LogMiningBig.py  # .. is a relative path, meaning one level up
```

**Do you see a problem with the memory configuration in this .sh file?** Whoever submitted this .sh file was asking ShARC to assign them 2 cores and 8GB per core. At the same time, their Spark job was asking for 10GB for the driver node. The obvious problem here is that there will not be any node with 10GB available to be set as a driver node since all nodes requested from ShARC will have a maximum of 8G available.

### Other configuration changes

Other configuration properties that we might find useful to change dynamically are `executor-memory` and `master local`. By default, `executor-memory` is 1GB, which might not be enough in some large data applications. You can change the `executor-memory` when using spark-submit, for example

```sh
spark-submit --driver-memory 10g --executor-memory 10g ../Code/LogMiningBig.py  
```

Just as before, one needs to be careful that the amount of memory dynamically requested through spark-submit does not go beyond what was requested from ShARC. In the past, we have seen .sh files intended to be run in one of our `rse-com6012` nodes with the following configuration

```sh
#!/bin/bash
#$ -l h_rt=6:00:00  
#$ -pe smp 10 
#$ -l rmem=20G 
#$ -o ../Output/Output.txt  
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M youremail@shef.ac.uk 
#$ -m ea #Email you when it finished or aborted
#$ -cwd 

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit --driver-memory 20g --executor-memory 30g ../Code/LogMiningBig.py  
```

**Do you see a problem with the memory configuration in this .sh file?** This script is asking ShARC for each core to have 20GB. This is fine because the scripts is requesting 200GB in total (10 times 20GB) which is lower than the maximum of 768GB. However, although spark-submit is requesting the same amount of 20GB per node for the `driver-memory`, the `executor-memory` is asking for 30G. There will not be any core with a real memory of 30G so the `executor-memory` request needs to be a maximum of 20G.

Another property that may be useful to change dynamically is `--master local`. So far, we have set the number of nodes in the `SparkSession.builder` inside the python script, for example,

```python
spark = SparkSession.builder \
    .master("local[2]") \
    .appName("COM6012 Spark Intro") \
    .config("spark.local.dir","/fastdata/YOUR_USERNAME") \
    .getOrCreate()
```

But we can also specify the number of cores in spark-submit using

```sh
spark-submit --driver-memory 5g --executor-memory 5g --master local[10] ../Code/LogMiningBig.py  
```

It is important to notice, however, that if `master local` is specified in spark-submit, you would need to remove that configuration from the SparkSession.builder,

```python
spark = SparkSession.builder \
    .appName("COM6012 Spark Intro") \
    .config("spark.local.dir","/fastdata/YOUR_USERNAME") \
    .getOrCreate()
```

Or make sure the number of cores you specify with SparkSession.builder matches the number of cores you specify when using spark-submit, 

```python
spark = SparkSession.builder \
    .master("local[10]") \
    .appName("COM6012 Spark Intro") \
    .config("spark.local.dir","/fastdata/YOUR_USERNAME") \
    .getOrCreate()
```

What happens if this is not the case? I.e. if the number of cores specified in spark-submit is different to the number of cores specified in SparkSession. In the past, we have seen the following instruction in the .sh file

```sh
spark-submit --driver-memory 5g --executor-memory 5g --master local[5] ../Code/LogMiningBig.py  
```

and when inspecting the python file, the following instruction for SparkSession

```python
spark = SparkSession.builder \
    .master("local[2]") \
    .appName("COM6012 Spark Intro") \
    .config("spark.local.dir","/fastdata/YOUR_USERNAME") \
    .getOrCreate()
```

**Do you see a problem with the number of cores in the configuration for these two files?** While spark-submit is requesting 5 cores, the SparkSession is requesting 2 cores. According to Spark documentation "Properties set directly on the SparkConf take highest precedence, then flags passed to spark-submit or spark-shell, then options in the spark-defaults.conf file." (see [this link](https://spark.apache.org/docs/3.2.1/configuration.html#application-properties)) meaning that the job will run with 2 cores and no 5 cores as intended in spark-submit.

Finally, the number of cores requested through spark-submit needs to match the number of cores requested from ShARC with `#$ -pe smp` in the .sh file. In the past, we have seen the following instruction in the .sh file

```sh
#!/bin/bash
#$ -l h_rt=6:00:00  
#$ -pe smp 10 
#$ -l rmem=20G 
#$ -o ../Output/Output.txt  
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M youremail@shef.ac.uk 
#$ -m ea #Email you when it finished or aborted
#$ -cwd 

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit --driver-memory 20g --executor-memory 20g --master local[15] ../Code/LogMiningBig.py 
```

with the corresponding python file including

```python
spark = SparkSession.builder \
    .master("local[15]") \
    .appName("COM6012 Spark Intro") \
    .config("spark.local.dir","/fastdata/YOUR_USERNAME") \
    .getOrCreate()
```

**Do you see a problem with the number of cores in the configuration for these two files?** Although the number of nodes requested through spark-submit and the SparkSession.builder are the same, that number does not match the number of cores requested to ShARC in the .sh file. Actually, spark-submit is requesting a higher number of nodes to the ones that could potentially be assigned by ShARC.

#### To change more configurations

You may search for example usage, an example that we used in the past **for very big data** is here for your reference only:

```sh
spark-submit --driver-memory 20g --executor-memory 20g --master local[10] --local.dir /fastdata/USERNAME --conf spark.driver.maxResultSize=4g test.py
```

**Observations**
1. If the real memory usage of your job exceeds `-l rmem=xxG` multiplied by the number of cores / nodes you requested then your job will be killed (see the [HPC documentation](https://docs.hpc.shef.ac.uk/en/latest/hpc/scheduler/index.html#interactive-jobs)).
2. A reminder that the more resources you request to ShARC, the longer you need to wait for them to become available to you.

## 2. Logistic regression in PySpark

We start with the [Spambase Dataset](http://archive.ics.uci.edu/ml/datasets/Spambase). We load the dataset and the names of the features and label. We cache the dataframe for efficiently performing several operations to rawdata inside a loop.

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

**Logistic regression** We are now in a position to train the logistic regression model. But before, let us look at a list of relevant parameters. A comprehensive list of parameters for [LogisticRegression](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.LogisticRegression.html) can be found in the Python API for PySpark.

> **maxIter**: max number of iterations. <p>
    **regParam**: regularization parameter ($\ge 0$).<p>
        **elasticNetParam**: mixing parameter for ElasticNet. It takes values in the range [0,1]. For $\alpha=0$, the penalty is an $\ell_2$. For $\alpha=1$, the penalty is an $\ell_1$.<p>
        **family**: binomial (binary classification) or multinomial (multi-class classification). It can also be 'auto'.<p>
            **standardization**: whether to standardize the training features before fitting the model. It can be true or false (True by default).

The function to optimise has the form

$$f(\mathbf{w}) = LL(\mathbf{w}) + \lambda\Big[\alpha\|\mathbf{w}\|_1 + (1-\alpha)\frac{1}{2}\|\mathbf{w}\|_2\Big]$$

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

    Accuracy = 0.925362 

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

This last result is consistent with the most relevant feature given by the [Decision Tree Classifier of Lab 3](https://github.com/haipinglu/ScalableML/blob/master/Lab%203%20-%20Scalable%20Decision%20trees.md).

A useful method for the logistic regression model is the [summary](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.LogisticRegressionSummary.html) method.

```python
lrModel1 = pipelineModellrL1.stages[-1]
lrModel1.summary.accuracy
```

    0.9111922141119222

The accuracy here is different to the one we got before. Why?

Other quantities that can be obtained from the summary include falsePositiveRateByLabel, precisionByLabel, recallByLabel, among others. For an exhaustive list, please read [here](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.LogisticRegressionSummary.html).

```python
print("Precision by label:")
for i, prec in enumerate(lrModel1.summary.precisionByLabel):
    print("label %d: %s" % (i, prec))
```

    Precision by label:
    label 0: 0.8979686057248384
    label 1: 0.9367201426024956

## 3. Exercises

**Note**: A *reference* solution will be provided in Blackboard for this part by the following Thursday (the latest).

### Exercise 1

Try a pure L2 regularisation and an elastic net regularisation on the same data partitions from above. Compare accuracies and find the most relevant features for both cases. Are these features the same than the one obtained for L1 regularisation?

### Exercise 2

Instead of creating a logistic regression model trying one type of regularisation at a time, create a [ParamGridBuilder](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.ParamGridBuilder.html) to be used inside a [CrossValidator](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.CrossValidator.html) to fine tune the best type of regularisation and the best parameters for that type of regularisation. Use five folds for the CrossValidator.

## 4. Additional exercise (optional)

**Note**: NO solutions will be provided for this part.

Create a logistic regression classifier that runs on the [default of credit cards](http://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) dataset. Several of the features in this dataset are categorical. Use the tools provided by PySpark (pyspark.ml.feature) for treating categorical variables.

Note also that this dataset has a different format to the Spambase dataset above - you will need to convert from XLS format to, say, CSV, before using the data. You can use any available tool for this: for example, Excell has an export option, or there is a command line tool <tt>xls2csv</tt> available on Linux.
