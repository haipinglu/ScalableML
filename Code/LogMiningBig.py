from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("local[2]") \
    .appName("COM6012 Spark Intro") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

logFile=spark.read.text("Data/NASA_access_log_Aug95.gz")
hostsJapan = logFile.filter(logFile.value.contains(".uk")).count()

print("\n\nHello Spark: There are %i hosts from UK.\n\n" % (hostsJapan))

spark.stop()
