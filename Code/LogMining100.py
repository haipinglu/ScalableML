from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("local[2]") \
    .appName("COM6012 Spark Intro") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

logFile=spark.read.text("Data/NASA_Aug95_100.txt")
hostsJapan = logFile.filter(logFile.value.contains(".jp")).count()

print("\n\nHello Spark: There are %i hosts from Japan.\n\n" % (hostsJapan))

spark.stop()
