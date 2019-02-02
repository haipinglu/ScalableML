from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("local[2]") \
    .appName("COM6012 Spark Intro") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

logFile=spark.read.text("../Data/NASA_access_log_Aug95.gz").cache()

hostsJapan = logFile.filter(logFile.value.contains(".jp")).count()
hostsUK = logFile.filter(logFile.value.contains(".uk")).count()

print("\n\nHello Spark: There are %i hosts from UK.\n" % (hostsUK))
print("Hello Spark: There are %i hosts from Japan.\n\n" % (hostsJapan))

spark.stop()
