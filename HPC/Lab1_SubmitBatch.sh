#!/bin/bash
#$ -l h_rt=6:00:00  # time needed in hours:mins:secs
#$ -pe smp 2 # number of cores requested
#$ -l rmem=8G # size of memory requested
#$ -o ../Output/COM6012_Lab1.txt  # This is where your output and errors are logged
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M youremail@shef.ac.uk # notify you by email, remove this line if you don't want to be notified
#$ -m ea # email you when it finished or aborted
#$ -cwd # run job from current directory

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit ../Code/LogMiningBig.py  # .. is a relative path, meaning one level up
