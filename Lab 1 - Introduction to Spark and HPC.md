# Lab 1 - Introduction to (Py)Spark and (Sheffield)HPC

[COM6012 Scalable Machine Learning **2021**](https://github.com/haipinglu/ScalableML) by [Haiping Lu](http://staffwww.dcs.shef.ac.uk/people/H.Lu/) at The University of Sheffield

## Study schedule

- [Task 1](#1-connect-to-hpc-and-install-spark): To finish by Wednesday. **Critical**
- [Task 2](#2-run-pyspark): To finish by Wednesday. **Critical**
- [Task 3](#3-log-mining-with-spark---example): To finish by Thursday. **Essential**
- [Task 4](#4-big-data-log-mining-with-spark): To finish by Thursday. **Essential**
- [Task 5](#5-exercises-reference-solutions-will-be-provided-on-the-following-monday): To finish before the next Monday. ***Exercise***
- [Task 6](#6-additional-ideas-to-explore-optional-no-solutions-will-be-provided): To explore further. *Optional*

**Suggested reading**:

- [Spark Overview](https://spark.apache.org/docs/latest/index.html)
- [Spark Quick Start](https://spark.apache.org/docs/3.0.1/quick-start.html) (Choose **Python** rather than the default *Scala*)
- Chapters 2 to 4 of [PySpark tutorial](https://runawayhorse001.github.io/LearningApacheSpark/pyspark.pdf) (several sections in Chapter 3 can be safely skipped)
- Reference: [PySpark documentation](https://spark.apache.org/docs/latest/api/python/index.html#)
- Reference: [PySpark source code](https://github.com/apache/spark/tree/master/python)

**Note - Please READ before proceeding**:

- HPC nodes are **shared** resources (**like buses/trains**) relying on considerate usage of every user. When requesting resources, if you ask for too much (e.g. 50 cores), it will take a long time to get allocated, particularly during "*rush hours*" (e.g. close to deadlines) and once allocated, it will leave much less for the others. If everybody is asking for too much, the system won't work and everyone suffers.
- We have five nodes (each with 40 cores, 768GB RAM) reserved for this module. You can specify `-P rse-com6012` (e.g. after `qrshx`) to get access. However, these nodes are not always more available, e.g. if all of us are using it. There are **100+** regular nodes, many of which may be idle.
- Please follow **all steps (step by step without skipping)** unless you are very confident in handling problems by yourself.
- Please try your best to follow the **study schedule** above to finish the tasks on time. If you start early/on time, you will find your problems early so that you can make good use of the online sessions to get help from the instructors and teaching assistants to fix your problems early, rather than getting panic close to an assessment deadline. Based on our experience from the past four years, rushing towards an assessment deadline in this module is likely to make you fall, sometimes painfully.

## 1. Connect to HPC and Install Spark

You **MUST** first connect to the [university's VPN](https://www.sheffield.ac.uk/it-services/vpn) unless you are on the campus network, which is unlikely during the lockdown.

### 1.1 Connect to ShARC HPC via SSH

Follow the [official instruction](https://docs.hpc.shef.ac.uk/en/latest/hpc/index.html) from our university. I have get your HPC account created already (first step done). Use your university **username** such as `abc18de` and the associated password to log in. You are requried to use [Multi-factor authentication (MFA)](https://www.sheffield.ac.uk/it-services/vpn) to connect to VPN. If you have problem logging in, please contact [it-servicedesk@sheffield.ac.uk](mailto:it-servicedesk@sheffield.ac.uk).

If you still have problem logging in, email me 1) your username; 2) a connection status of your VPN, 3) a screen capture of the error message (**never send me your password**). I will try to see whether I identify what the problem is. If you can log in but encounter HPC-related problem, you can email [hpc@sheffield.ac.uk](hpc@sheffield.ac.uk) for help.

Following the [official instructions](https://docs.hpc.shef.ac.uk/en/latest/hpc/connecting.html) for [Windows](https://docs.hpc.shef.ac.uk/en/latest/hpc/connecting.html#ssh-client-software-on-windows) or [Mac OS/X and Linux](https://docs.hpc.shef.ac.uk/en/latest/hpc/connecting.html#ssh-client-software-on-mac-os-x-and-linux) to open a terminal and connect to sharc via SSH by

```sh
ssh -X $USER@sharc.shef.ac.uk  # -X is optional as we do not show graphics directly so simply ssh $USER@sharc.shef.ac.uk will do
```

You need to replace `$USER` with your username, let's assume it is `abc1de`. If successful, you should see 

`[abc1de@sharc-login1 ~]$`

`abc1de` should be your username. 

#### MobaXterm tips

- You can save the host, username (and password if your computer is secure) as a **Session** if you want to save time in future.
- Shift + Insert for copy (after text selection), not Ctrl + C. Ctrl + V works as paste (to confirm on new PC).
- You can drag your file or folder to the left directory pane of MobaXterm.

### 1.2 Set up the environment and install PySpark

#### Start an interactive session

Type `qrshx` for a *regular- node **or** `qrshx -P rse-com6012` for a com6012-reserved node

#### Load Java and conda

`module load apps/java/jdk1.8.0_102/binary`

`module load apps/python/conda`

#### Create a virtual environment called `myspark`

`conda create -n myspark python=3.6`

When you are asked whether to proceed, say `y`

#### Activate the environment

`source activate myspark`

You **must** see `(myspark) [abc1de@sharc-nodeXXX ~]$`, i.e. **(myspark)** in front, before proceeding. Otherwise, you did not get the proper environment. Check the above steps. 

#### Install pyspark 3.0.1 using `pip`

`pip install pyspark==3.0.1`

When you are asked whether to proceed, say `y`

#### Run pyspark

`pyspark`

You should see spark version **3.0.1** displayed like below

```sh
......
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /__ / .__/\_,_/_/ /_/\_\   version 3.0.1
      /_/
Using Python version 3.6.2 (default, Jul 20 2017 13:51:32)
SparkSession available as 'spark'.
>>>
```

**Bingo!** Now you are in pyspark! Quit pyspark shell by `Ctrl + D`.

### 1.3 Get more familiar with the HPC

**Terminal/command line**: learn the [basic use of the command line](https://github.com/mikecroucher/Intro_to_HPC/blob/gh-pages/terminal_tutorial.md) in Linux, e.g. use `pwd` to find out your **current directory**.

**Transfer files**: learn how to [transfer files to/from ShARC HPC](https://www.sheffield.ac.uk/it-services/research/hpc/using/access).

**Line ending warning**: if you are using Windows, you should be aware that [line endings differ between Windows and Linux](https://stackoverflow.com/questions/426397/do-line-endings-differ-between-windows-and-linux). If you edit a shell script (below) in Windows, make sure that you use a Unix/Linux compatible editor or do the conversion before using it on HPC.

### 1.4 *Optional: Install PySpark on your own machine-  

This module focuses on the HPC terminal. You are expected to use the HPC terminal to complete the labs. ALL assessments use the HPC terminal.

Installation of PySpark on your own machine is more complicated than installing a regular python library because it depends on Java (i.e. not pure python). The following steps are typically needed:

- Install [**Java 8**](https://www.java.com/en/download/manual.jsp), i.e. java version *1.8.xxx*. Most instructions online ask you to install *Java SDK*, which is heavier. *Java JRE- is lighter and sufficient for pyspark.
- Install Python **3.6+** (if not yet)
- Install PySpark **3.0.1** with **Hadoop 2.7**
- Set up the proper environments (see references below)

As far as I know, it is not necessary to install *Scala*.

Different OS (Windows/Linux/Mac) may have different problems. We provide some references below if you wish to try but it is *not required- and we can provide only very limited support on this task (i.e. we may not be able to solve all problems that you may encounter).

If you do want to install PySpark and run Jupyter Notebooks on your own machine, you need to complete the steps above with reference to the instructions below for your OS (Windows/Linux/Mac).

#### References (use with caution, not necessarily up to date or the best)

If you follow the steps in these references, be aware that they are not up to date so you should install the correct versions: **Java 1.8**, Python **3.6+**, PySpark **3.0.1** with **Hadoop 2.7**. *Scala- is optional.

- Windows: 1) [Install Spark on Windows (PySpark)](https://medium.com/@GalarnykMichael/install-spark-on-windows-pyspark-4498a5d8d66c) (with video) 2) [How to install Spark on Windows in 5 steps](https://medium.com/@dvainrub/how-to-install-apache-spark-2-x-in-your-pc-e2047246ffc3).

- Linux: 1) [Install PySpark on Ubuntu](https://medium.com/@GalarnykMichael/install-spark-on-ubuntu-pyspark-231c45677de0) (with video); 2)[Installing PySpark with JAVA 8 on ubuntu 18.04](https://towardsdatascience.com/installing-pyspark-with-java-8-on-ubuntu-18-04-6a9dea915b5b)

- Mac: 1) [Install Spark on Mac (PySpark)](https://medium.com/@GalarnykMichael/install-spark-on-mac-pyspark-453f395f240b) (with video); 2) [Install Spark/PySpark on Mac](https://medium.com/@yajieli/installing-spark-pyspark-on-mac-and-fix-of-some-common-errors-355a9050f735)

#### Install PySpark on Windows

Here we provide detailed instructions only for Windows.

- Install Java
  - Download `jre-8u...` and install [**Java 8 JRE**](https://www.java.com/en/download/manual.jsp).
  - Find the path for the installed Java under `Program files\Java\jre1.8.0_xxx` (replace `xxx` with the number you see) and set two environment variables to know where to find Java:
    - `JAVA_HOME = C:\Progra~1\Java\jdk1.8.0_xxx`
    - `PATH += C:\Progra~1\Java\jdk1.8.0_xxx\bin`
  - Check: open a command prompt and type `java -version`. If you can see the version displayed, congratulations. Otherwise, check the above.
- Install Python
  - Install [Python 3.6+](https://www.python.org/downloads/). Open a command and type `python --version` to check your version to be 3.6+.
- Install PySpark (Alternatively, you may try `pip install pyspark==3.0.1`)
  - Download Spark **3.0.1** for Hadoop **2.7**, i.e. `spark-3.0.1-bin-hadoop2.7.tgz`.
  - Extract the `.tgz` file (e.g. using 7zip) into `C:\Spark` so that extracted files are at `C:\Spark\spark-3.0.1-bin-hadoop2.7`.
  - Set the environment variables: 
    - `SPARK_HOME = C:\Spark\spark-3.0.1-bin-hadoop2.7`
    - `PATH += C:\Spark\spark-3.0.1-bin-hadoop2.7\bin`
  - Download [**winutils.exe** for hadoop 2.7](https://github.com/steveloughran/winutils/blob/master/hadoop-2.7.1/bin/winutils.exe) and move it to `C:\Spark\spark-3.0.1-bin-hadoop2.7\bin`
  - Set the environment variable:
    - `HADOOP_HOME = C:\Spark\spark-3.0.1-bin-hadoop2.7`
    - `PYTHONPATH = %SPARK_HOME%\python;%SPARK_HOME%\python\lib\py4j-<version>-src.zip;%PYTHONPATH%` (just check what py4j version you have in your `spark/python/lib` folder to replace `<version>` ([source](https://stackoverflow.com/questions/53161939/pyspark-error-does-not-exist-in-the-jvm-error-when-initializing-sparkcontext?noredirect=1&lq=1)).

Now open a command prompt and type `pyspark`. You should see pyspark 3.0.1 running as above.

*Known issue on Windows* There may be a `ProcfsMetricsGetter` warning. If you press `Enter`, the warning will disappear. I did not find a better solution to get rid of it. It does not seem harmful either. If you know how to deal with it. Please let me know. Thanks. [Reference 1](https://stackoverflow.com/questions/63762106/rn-procfsmetricsgetter-exception-when-trying-to-compute-pagesize-as-a-result-r); [Reference 2](https://stackoverflow.com/questions/60257377/encountering-warn-procfsmetricsgetter-exception-when-trying-to-compute-pagesi); [Reference 3](https://stackoverflow.com/questions/61863127/getting-error-while-setting-pyspark-environment).

**From this point on, we will assume that you are using the HPC terminal unless otherwise stated**. Run PySpark shell on your own machine can do the same job.

## 2. Run PySpark

Once PySpark has been installed, after _each_ log-in, you need to do the following to run PySpark.

### Get a node and activate myspark

- Get a node via `qrshx` or `qrshx -P rse-com6012`.
- Activate the environment by

   ```sh
   module load apps/java/jdk1.8.0_102/binary
   module load apps/python/conda
   source activate myspark
  ```

  Alternatively, put `HPC/myspark.sh` under your *root* directory (see above on how to transfer files) and do `source myspark.sh` will run these three commands in sequence. You could modify it further to suit yourself better.

### Interactive shell

Run pyspark (optionally, specify to use multiple cores):

```sh
pyspark  # pyspark --master local[4] for 4 cores
```

You will see the spark splash above. `spark` ([SparkSession](https://spark.apache.org/docs/2.1.0/api/python/pyspark.sql.html?highlight=sparksession#pyspark.sql.SparkSession)) and `sc` ([SparkContext](https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.SparkContext)) are automatically created.

Check your SparkSession and SparkContext object and you will see something like

```python
>>> spark
<pyspark.sql.session.SparkSession object at 0x2b3a2ad4c630>
>>> sc
<SparkContext master=local[*] appName=PySparkShell>
```

Let us do some simple computing (squares)

```python
>>> nums = sc.parallelize([1,2,3,4])
>>> nums.map(lambda x: x*x).collect()
[1, 4, 9, 16]
```

## 3. Log Mining with Spark - Example

This example deals with **Semi-Structured** data in a text file.

Firstly, you need to **make sure the file is in the proper directory and change the file path if necessary**, on either HPC or local machine, e.g. using ``pwd` to see the current directly, `ls` (or `dir` in Windows) to see the content. Also review how to [**transfer files to HPC**](https://www.sheffield.ac.uk/cics/research/hpc/using/access) and [MobaXterm tips](#MobaXterm-tips) for Windows users.

Now quit pyspark by `Ctrl + D`. Take a look at where you are

```sh
(myspark) [abc1de@sharc-node175 ~]$ pwd
/home/abc1de
```

`abc1de` should be your username. Let us make a new directory called `com6012` and go to it

```sh
mkdir com6012
cd com6012
```

Let us make a copy of our teaching materials at this directory via

```sh
git clone --depth 1 https://github.com/haipinglu/ScalableML
```

If `ScalableML` is not empty (e.g. you have cloned a copy already), this will give you an error. You need to delete the cloned version (the whole folder) via `rm -rf ScalableML`. Be careful that you can **NOT** undo this delete so make sure **you do not have anything valuable (e.g. your assignment) there** if you do this delete. 

You are advised to create a **separate folder** for your own work under `com6012`, e.g. `mywork`.

Let us check

```sh
(myspark) [abc1de@sharc-node175 com6012]$ ls
ScalableML
(myspark) [abc1de@sharc-node175 com6012]$ cd ScalableML
(myspark) [abc1de@sharc-node175 ScalableML]$ ls
Code  Data  HPC  README.md  Slides
(myspark) [abc1de@sharc-node175 ScalableML]$ pwd
/home/abc1de/com6012/ScalableML
```

You can see that files on the GitHub has been downloaded to your HPC directory `/home/abc1de/com6012/ScalableML`. Now start spark shell by `pyspark` (again you should see the splash) and now we 

- read the log file `NASA_Aug95_100.txt` under the folder `Data`
- count the number of lines
- take a look at the first line

```python
>>> logFile=spark.read.text("Data/NASA_Aug95_100.txt")
>>> logFile
DataFrame[value: string]
>>> logFile.count()
100
>>> logFile.first()
Row(value='in24.inetnebr.com - - [01/Aug/1995:00:00:01 -0400] "GET /shuttle/missions/sts-68/news/sts-68-mcc-05.txt HTTP/1.0" 200 1839')
```

You may open the text file to verify than pyspark is doing the right things.

**Question**: How many accesses are from Japan?

Now suppose you are asked to answer the question above. What do you need to do?

- Find those logs from Japan (by IP domain `.jp`)
- Show the first 5 logs to check whether you are getting what you want.

```python
>>> hostsJapan = logFile.filter(logFile.value.contains(".jp"))
>>> hostsJapan.show(5,False)
+--------------------------------------------------------------------------------------------------------------+
|value                                                                                                         |
+--------------------------------------------------------------------------------------------------------------+
|kgtyk4.kj.yamagata-u.ac.jp - - [01/Aug/1995:00:00:17 -0400] "GET / HTTP/1.0" 200 7280                         |
|kgtyk4.kj.yamagata-u.ac.jp - - [01/Aug/1995:00:00:18 -0400] "GET /images/ksclogo-medium.gif HTTP/1.0" 200 5866|
|kgtyk4.kj.yamagata-u.ac.jp - - [01/Aug/1995:00:00:21 -0400] "GET /images/NASA-logosmall.gif HTTP/1.0" 304 0   |
|kgtyk4.kj.yamagata-u.ac.jp - - [01/Aug/1995:00:00:21 -0400] "GET /images/MOSAIC-logosmall.gif HTTP/1.0" 304 0 |
|kgtyk4.kj.yamagata-u.ac.jp - - [01/Aug/1995:00:00:22 -0400] "GET /images/USA-logosmall.gif HTTP/1.0" 304 0    |
+--------------------------------------------------------------------------------------------------------------+
only showing top 5 rows

>>> hostsJapan.count()
11
```

Now you have used pyspark for some (very) simple data analytic task.

### Self-contained Application

To run a self-contained application, you need to **exit your shell, by `Ctrl+D` first**.

Create a file `LogMining100.py`

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("local[2]") \
    .appName("COM6012 Spark Intro") \
    .config("spark.local.dir","/fastdata/YOUR_USERNAME") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")  # This can only affect the log level after it is executed.

logFile=spark.read.text("Data/NASA_Aug95_100.txt").cache()
hostsJapan = logFile.filter(logFile.value.contains(".jp")).count()

print("\n\nHello Spark: There are %i hosts from Japan.\n\n" % (hostsJapan))

spark.stop()
```

Change `YOUR_USERNAME` in `/fastdata/YOUR_USERNAME` to your username. If you are running on your local machine, change `/fastdata/YOUR_USERNAME` to a temporal directory such as `C:\temp`.

Actually the file has been created for you under the folder `Code` so you can just run it

```sh
spark-submit Code/LogMining100.py
```

You will see lots of logging info output such as
```sh
21/02/05 00:35:57 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
21/02/05 00:35:59 INFO SparkContext: Running Spark version 3.0.1
.....................
21/02/05 00:35:59 INFO ResourceUtils: Resources for spark.driver:

21/02/05 00:35:59 INFO ResourceUtils: ==============================================================
21/02/05 00:35:59 INFO SparkContext: Submitted application: COM6012 Spark Intro
.....................
21/02/05 00:36:03 INFO SharedState: Warehouse path is 'file:/home/abc1de/com6012/ScalableML/spark-warehouse'.


Hello Spark: There are 11 hosts from Japan.

```

The output is verbose so I did not show all (see `Output/COM6012_Lab1_SAMPLE.txt` for the verbose output example). We can set the log level easily after `sparkContext` is created but not before (it is a bit complicated). I leave two blank lines before printing the result so it is early to see.

## 4. Big Data Log Mining with Spark

**Data**: Download the August data in gzip (NASA_access_log_Aug95.gz) from [NASA HTTP server access log](http://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html) (this file is uploaded to `ScalableML/Data` if you have problems downloading, so actually it is already downloaded on your HPC earlier) and put into your `Data` folder. `NASA_Aug95_100.txt` above is the first 100 lines of the August data.

**Question**: How many accesses are from Japan and UK respectively?

Create a file `LogMiningBig.py`

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("local[2]") \
    .appName("COM6012 Spark Intro") \
    .config("spark.local.dir","/fastdata/YOUR_USERNAME") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")  # This can only affect the log level after it is executed.

logFile=spark.read.text("../Data/NASA_access_log_Aug95.gz").cache()

hostsJapan = logFile.filter(logFile.value.contains(".jp")).count()
hostsUK = logFile.filter(logFile.value.contains(".uk")).count()

print("\n\nHello Spark: There are %i hosts from UK.\n" % (hostsUK))
print("Hello Spark: There are %i hosts from Japan.\n\n" % (hostsJapan))

spark.stop()
```

**Spark can read gzip file directly**. You do not need to unzip it to a big file. Also note the use of **cache()** above.

### Run a program in batch mode

See [how to submit batch jobs to ShARC](https://docs.hpc.shef.ac.uk/en/latest/hpc/scheduler/submit.html#running-batch-jobs) and follow the instructions for **SGE**. **Reminder:** The more resources you request, the longer you need to queue.

Interactive mode will be good for learning, exploring and debugging, with smaller data. For big data, it will be more convenient to use batch processing. You submit the job to the node to join a queue. Once allocated, your job will run, with output properly recorded. This is done via a shell script. **Warning: Do not create such a file under WINDOWS text format** (see the *line ending warning* above).

Create a file `Lab1_SubmitBatch.sh`

```sh
#!/bin/bash
#$ -l h_rt=6:00:00  #time needed
#$ -pe smp 2 #number of cores
#$ -l rmem=8G #number of memery
#$ -o ../Output/COM6012_Lab1.txt  #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M youremail@shef.ac.uk #Notify you by email, remove this line if you don't like
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit ../Code/LogMiningBig.py  # .. is a relative path, meaning one level up
```

- Get necessary files on your ShARC.
- Start a session with command `qrshx`.
- Go to the `HPC` directory to submit your job via the `qsub` command (can be run at the **login node**).
- The output file will be under `HPC`.

```sh
cd HPC
qsub Lab1_SubmitBatch.sh # or qsub HPC/Lab1_SubmitBatch.sh if you are at /home/abc1de/com6012/ScalableML
```

Check the status of your queuing/running job(s) `qstat` (jobs not shown are finished already).

Check your output file, which is **`COM6012_Lab1.txt`** in the `Output` folder specified with option **`-o`** above. You can change it to a name you like. A sample output file named `COM6012_Lab1_SAMPLE.txt` is now in the GitHub `Output` folder. The results are

```sh
Hello Spark: There are 35924 hosts from UK.

Hello Spark: There are 71600 hosts from Japan.
```

## 5. Exercises (reference solutions will be provided on the following Monday)

The analytic task you are doing above is *Log Mining*. You can imaging nowadays, log files are big and manual analysis will be time consuming. Follow examples above, answer the following questions on **NASA_access_log_Aug95.gz**.

1. How many requests are there in total?
2. How many requests are from `gateway.timken.com`?
3. How many requests are on 15th August 1995?
4. How many 404 (page not found) errors are there in total?
5. How many 404 (page not found) errors are there on 15th August?
6. How many 404 (page not found) errors from `gateway.timken.com` are there on 15th August?

You are encouraged to try out in the pyspark shell first to figure out the right solutions and then write a Python script, e.g. `Lab1_exercise.py` with a batch file (e.g. `Lab1_Exercise_Batch.sh` to produce the output neatly under `Output`, e.g. in a file `Lab1_exercise.txt`.

## 6. Additional ideas to explore (*optional*, NO solutions will be provided)

### More log mining questions

You are encouraged to explore these more challenging questions by consulting the [`pyspark.sql` APIs](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html) to learn more. We will not provide solutions but Session 2 will make answering these questions easier.

- How many **unique** hosts on a particular day (e.g., 15th August)?
- How many **unique** hosts in total (i.e., in August 1995)?
- Which host is the most frequent visitor?
- How many different types of return codes?
- How many requests per day on average?
- How many requests per host on average?
- Any other question that you (or your **imagined clients**) are interested in to find out.

### The effects of caching

- **Compare** the time taken to complete your jobs **with and without** `cache()`.

### The effects of the number of cores

- **Compare** the time taken to complete your jobs with 2, 4, 8, 16, and 32 cores.

## 7. Acknowledgements

Many thanks to Twin, Will, Mike, Vamsi for their kind help and all those kind contributors of open resources.

The log mining problem is adapted from [UC Berkeley cs105x L3](https://www.edx.org/course/introduction-apache-spark-uc-berkeleyx-cs105x).
