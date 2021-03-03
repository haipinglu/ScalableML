# Lab 5: PCA for dimensionality reduction and Spark data types

[COM6012 Scalable Machine Learning **2021**](https://github.com/haipinglu/ScalableML) by [Haiping Lu](http://staffwww.dcs.shef.ac.uk/people/H.Lu/) at The University of Sheffield

## Study schedule

- [Task 1](#1-): To finish by Wednesday. **Essential**
- [Task 2](#2-): To finish by Thursday. **Essential**
- [Task 3](#3-exercises): To finish before the next Monday. ***Exercise***
- [Task 4](#4-additional-ideas-to-explore-optional): To explore further. *Optional*

### Suggested reading

- [Extracting, transforming and selecting features](https://spark.apache.org/docs/3.0.1/ml-features.html)
- [PCA in Spark DataFrame API `pyspark.ml`](https://spark.apache.org/docs/3.0.1/ml-features.html#pca)
- [SVD in Spark RDD API `pyspark.mllib`](https://spark.apache.org/docs/3.0.1/mllib-dimensionality-reduction.html#singular-value-decomposition-svd)
- [StandardScaler in Spark](https://spark.apache.org/docs/3.0.1/ml-features.html#standardscaler) to standardise/normalise data to unit standard deviation and/or zero mean.
- [Data Types - RDD-based API](https://spark.apache.org/docs/3.0.1/mllib-data-types.html)
- [PCA on Wiki](https://en.wikipedia.org/wiki/Principal_component_analysis)
- [Understanding Dimension Reduction with Principal Component Analysis (PCA)](https://blog.paperspace.com/dimension-reduction-with-principal-component-analysis/)
- [Principal Component Analysis explained on Kaggle](https://www.kaggle.com/nirajvermafcb/principal-component-analysis-explained) with data available [here](https://www.kaggle.com/liujiaqi/hr-comma-sepcsv), and background info [here](https://rstudio-pubs-static.s3.amazonaws.com/345463_37f54d1c948b4cdfa181541841e0db8a.html)

## 1. Data Types in RDD-based API

To deal with data efficiently, Spark considers different [data types](https://spark.apache.org/docs/3.0.1/mllib-data-types.html). In particular, MLlib supports local vectors and matrices stored on a single machine, as well as distributed matrices backed by one or more RDDs. Local vectors and local matrices are simple data models that serve as public interfaces. The underlying linear algebra operations are provided by [Breeze](http://www.scalanlp.org/). A training example used in supervised learning is called a “labeled point” in MLlib.

### [Local vector](https://spark.apache.org/docs/3.0.1/mllib-data-types.html#local-vector):  Dense vs Sparse

> A local vector has integer-typed and 0-based indices and double-typed values, stored on a single machine. MLlib supports two types of local vectors: dense and sparse. A dense vector is backed by a double array representing its entry values, while a sparse vector is backed by two parallel arrays: indices and values. For example, a vector (1.0, 0.0, 3.0) can be represented in dense format as [1.0, 0.0, 3.0] or in sparse format as (3, [0, 2], [1.0, 3.0]), where 3 is the size of the vector.

Check out the [Vector in RDD API](https://spark.apache.org/docs/3.0.1/api/python/pyspark.mllib.html#pyspark.mllib.linalg.Vectors) or [Vector in DataFrame API](https://spark.apache.org/docs/3.0.1/api/python/pyspark.ml.html#pyspark.ml.linalg.Vector) (see method `.Sparse()`) and [SparseVector in RDD API ](https://spark.apache.org/docs/3.0.1/api/python/pyspark.mllib.html#pyspark.mllib.linalg.SparseVector) or [SparseVector in DataFrame API ](https://spark.apache.org/docs/3.0.1/api/python/pyspark.ml.html#pyspark.ml.linalg.SparseVector). The official example is below

```python
import numpy as np
from pyspark.mllib.linalg import Vectors

dv1 = np.array([1.0, 0.0, 3.0])  # Use a NumPy array as a dense vector.
dv2 = [1.0, 0.0, 3.0]  # Use a Python list as a dense vector.
sv1 = Vectors.sparse(3, [0, 2], [1.0, 3.0])  # Create a SparseVector.
```

Note the vector created by `Vectors.sparse()` is of type `SparseVector()`

```python
sv1
# SparseVector(3, {0: 1.0, 2: 3.0})
```

To view the sparse vector in a dense format

```python
sv1.toArray()
# array([1., 0., 3.])
```

### [Labeled point](https://spark.apache.org/docs/3.0.1/mllib-data-types.html#labeled-point)

> A labeled point is a local vector, either dense or sparse, associated with a label/response. In MLlib, labeled points are used in supervised learning algorithms. We use a double to store a label, so we can use labeled points in both regression and classification. For binary classification, a label should be either 0 (negative) or 1 (positive). For multiclass classification, labels should be class indices starting from zero: 0, 1, 2, ....

See [LabeledPoint API in MLlib](https://spark.apache.org/docs/3.0.1/api/python/pyspark.mllib.html#pyspark.mllib.regression.LabeledPoint). Now, we create a labeled point with a positive label and a dense feature vector, as well as a labeled point with a negative label and a sparse feature vector.

```python
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint

pos = LabeledPoint(1.0, [1.0, 0.0, 3.0])
neg = LabeledPoint(0.0, SparseVector(3, [0, 2], [1.0, 3.0]))

neg
# LabeledPoint(0.0, (3,[0,2],[1.0,3.0]))
neg.label
# 0.0
neg.features
# SparseVector(3, {0: 1.0, 2: 3.0})
```

Now view the features as dense vector (rather than sparse vector)

```python
neg.features.toArray()
# array([1., 0., 3.])
```

### [Local matrix](https://spark.apache.org/docs/3.0.1/mllib-data-types.html#local-matrix)

> A local matrix has integer-typed row and column indices and double-typed values, stored on a single machine. MLlib supports dense matrices, whose entry values are stored in a single double array in column-major order, and sparse matrices, whose non-zero entry values are stored in the Compressed Sparse Column (CSC) format in column-major order. For example, we create a dense matrix ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)) and a sparse matrix ((9.0, 0.0), (0.0, 8.0), (0.0, 6.0)) in the following:

```python
from pyspark.mllib.linalg import Matrix, Matrices

dm2 = Matrices.dense(3, 2, [1, 3, 5, 2, 4, 6]) 
sm = Matrices.sparse(3, 2, [0, 1, 3], [0, 2, 1], [9, 6, 8])
print(dm2)
# DenseMatrix([[1., 2.],
#              [3., 4.],
#              [5., 6.]])
print(sm)
# 3 X 2 CSCMatrix
# (0,0) 9.0
# (2,1) 6.0
# (1,1) 8.0
```

Here the [compressed sparse column (CSC or CCS) format](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS)) is used for sparse matrix representation.
> values are read first by column, a row index is stored for each value, and column pointers are stored. For example, CSC is (val, row_ind, col_ptr), where val is an array of the (top-to-bottom, then left-to-right) non-zero values of the matrix; row_ind is the row indices corresponding to the values; and, col_ptr is the list of val indexes where each column starts. 


```python
dsm=sm.toDense()
print(dsm)
# DenseMatrix([[9., 0.],
#              [0., 8.],
#              [0., 6.]])
```

### [Distributed matrix](https://spark.apache.org/docs/3.0.1/mllib-data-types.html#distributed-matrix)

> A distributed matrix has long-typed row and column indices and double-typed values, stored distributively in one or more RDDs. It is very important to choose the right format to store large and distributed matrices. Converting a distributed matrix to a different format may require a global shuffle, which is quite expensive. Four types of distributed matrices have been implemented so far.

#### RowMatrix

> The basic type is called RowMatrix. A RowMatrix is a row-oriented distributed matrix without meaningful row indices, e.g., a collection of feature vectors. It is backed by an RDD of its rows, where each row is a local vector. We assume that the number of columns is not huge for a RowMatrix so that a single local vector can be reasonably communicated to the driver and can also be stored / operated on using a single node.
> Since each row is represented by a local vector, the number of columns is limited by the integer range but it should be much smaller in practice.

Now we create an RDD of vectors `rows`, from which we create a RowMatrix `mat`.

```python
from pyspark.mllib.linalg.distributed import RowMatrix

rows = sc.parallelize([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
mat = RowMatrix(rows)

m = mat.numRows()  # Get its size: m=4, n=3
n = mat.numCols()  

rowsRDD = mat.rows  # Get the rows as an RDD of vectors again.
```

We can view the RowMatrix in a dense matrix format

```python
rowsRDD.collect()
# [DenseVector([1.0, 2.0, 3.0]), DenseVector([4.0, 5.0, 6.0]), DenseVector([7.0, 8.0, 9.0]), DenseVector([10.0, 11.0, 12.0])]
```

## 2. PCA

[Principal component analysis](http://en.wikipedia.org/wiki/Principal_component_analysis) (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables (entities each of which takes on various numerical values) into a set of values of linearly uncorrelated variables called **principal components (PCs)**. A PCA class trains a model to project vectors to a low-dimensional space using PCA and this is probably the most commonly used **dimensionality reduction** method.

### PCA in DataFrame-based API `pyspark.ml`  

Check out the [API](https://spark.apache.org/docs/3.0.1/api/python/pyspark.ml.html#pyspark.ml.feature.PCA). Check [`pyspark.ml.feature.PCAModel`](https://spark.apache.org/docs/3.0.1/api/python/pyspark.ml.html#pyspark.ml.feature.PCAModel) too to see what is available for the fitted model. Let us project three 5-dimensional feature vectors into 2-dimensional principal components.

```python
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors

data = [(Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
        (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
        (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]
df = spark.createDataFrame(data, ["features"])
df.show()
# +--------------------+
# |            features|
# +--------------------+
# | (5,[1,3],[1.0,7.0])|
# |[2.0,0.0,3.0,4.0,...|
# |[4.0,0.0,0.0,6.0,...|
# +--------------------+

pca = PCA(k=2, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(df)

result = model.transform(df).select("pcaFeatures")
result.show(truncate=False)
# +----------------------------------------+
# |pcaFeatures                             |
# +----------------------------------------+
# |[1.6485728230883807,-4.013282700516296] |
# |[-4.645104331781534,-1.1167972663619026]|
# |[-6.428880535676489,-5.337951427775355] |
# +----------------------------------------+
```

Check the explained variance in percentage

```python
model.explainedVariance
# DenseVector([0.7944, 0.2056])
```

Take a look at the principal components Matrix. Each column is one principal component.

```python
 print(model.pc)
# DenseMatrix([[-0.44859172, -0.28423808],
#              [ 0.13301986, -0.05621156],
#              [-0.12523156,  0.76362648],
#              [ 0.21650757, -0.56529588],
#              [-0.84765129, -0.11560341]])
```

### PCA in RDD-based API `pyspark.mllib`

#### Eigendecomposition for PCA

`pyspark.mllib` supports PCA for **tall-and-skinny** (big $n$, small $d$) matrices stored in row-oriented format and any Vectors. We demonstrate how to compute principal components on a [<tt>RowMatrix</tt>](http://spark.apache.org/docs/3.0.1/mllib-data-types.html#rowmatrix) and use them to project the vectors into a low-dimensional space in the cell below.

```python
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix

rows = sc.parallelize([
    Vectors.sparse(5, {1: 1.0, 3: 7.0}),
    Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
    Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
])
rows.collect()
# [SparseVector(5, {1: 1.0, 3: 7.0}), DenseVector([2.0, 0.0, 3.0, 4.0, 5.0]), DenseVector([4.0, 0.0, 0.0, 6.0, 7.0])]

mat = RowMatrix(rows)
```

Compute the top 2 principal components, which are stored in a local dense matrix (the same as above).

```python
pc = mat.computePrincipalComponents(2)
print(pc)
# DenseMatrix([[-0.44859172, -0.28423808],
#              [ 0.13301986, -0.05621156],
#              [-0.12523156,  0.76362648],
#              [ 0.21650757, -0.56529588],
#              [-0.84765129, -0.11560341]])
```

Project the rows to the linear space spanned by the top 2 principal components (the same as above)

```python
projected = mat.multiply(pc)
projected.rows.collect()
# [DenseVector([1.6486, -4.0133]), DenseVector([-4.6451, -1.1168]), DenseVector([-6.4289, -5.338])]
```

Now we convert to dense rows to see the matrix

```python
from pyspark.mllib.linalg import DenseVector

denseRows = rows.map(lambda vector: DenseVector(vector.toArray()))
denseRows.collect()
# [DenseVector([0.0, 1.0, 0.0, 7.0, 0.0]), DenseVector([2.0, 0.0, 3.0, 4.0, 5.0]), DenseVector([4.0, 0.0, 0.0, 6.0, 7.0])]
```

#### SVD for PCA  - more *scalable* way to do PCA

Read [SVD in RDD-based API `pyspark.mllib`](https://spark.apache.org/docs/3.0.1/mllib-dimensionality-reduction.html#singular-value-decomposition-svd). As covered in the lecture, we will need SVD for PCA on large-scale data. Here, we use it on the same small toy example to examine the relationship with eigenvalue decomposition based PCA methods above.

We compute the top 2 singular values and corresponding singular vectors.

```python
svd = mat.computeSVD(2, computeU=True)
U = svd.U       # The U factor is a RowMatrix.
s = svd.s       # The singular values are stored in a local dense vector.
V = svd.V       # The V factor is a local dense matrix.
```

If we are doing it right, the **right** singular vectors should be the same as the eigenvectors.

```python
print(V)
print(V)
# DenseMatrix([[-0.31278534,  0.31167136],
#              [-0.02980145, -0.17133211],
#              [-0.12207248,  0.15256471],
#              [-0.71847899, -0.68096285],
#              [-0.60841059,  0.62170723]])
```

But it is **not the same**! Why? Remeber that we need to do **centering**! We can do so use the [StandardScaler (check out the API](https://spark.apache.org/docs/3.0.1/mllib-feature-extraction.html#standardscaler)) to center the data, i.e., remove the mean.

```python
from pyspark.mllib.feature import StandardScaler

standardizer = StandardScaler(True, False)
model = standardizer.fit(rows)
centeredRows = model.transform(rows)
centeredRows.collect()
# [DenseVector([-2.0, 0.6667, -1.0, 1.3333, -4.0]), DenseVector([0.0, -0.3333, 2.0, -1.6667, 1.0]), DenseVector([2.0, -0.3333, -1.0, 0.3333, 3.0])]
centeredmat = RowMatrix(centeredRows)
```

Compute the top 2 singular values and corresponding singular vectors.

```python
svd = centeredmat.computeSVD(2, computeU=True)
U = svd.U       # The U factor is a RowMatrix.
s = svd.s       # The singular values are stored in a local dense vector.
V = svd.V       # The V factor is a local dense matrix.
```

Check the **PC** obtained this time (it is the same as the above PCA methods now)

```python
print(V)
DenseMatrix([[-0.44859172, -0.28423808],
             [ 0.13301986, -0.05621156],
             [-0.12523156,  0.76362648],
             [ 0.21650757, -0.56529588],
             [-0.84765129, -0.11560341]])
```

Let us examine the relationships between the singular values and the eigenvalues.

```python
print(s)
# [6.001041088520536,3.0530049438580336]
```    

We get the eigenvalues by taking squares of the singular values

```python
evs=s*s
 print(evs)
[36.012494146111734,9.320839187221594]
```

Now we compute the percentage of variance captures and compare with the above to verify (see/search `model.explainedVariance`).

```python
evs/sum(evs)
# DenseVector([0.7944, 0.2056])
```

## 3. Exercises

### PCA on iris

Study the [Iris flower data set](https://en.wikipedia.org/wiki/Iris_flower_data_set) `iris.csv` under `Data` with PCA.

1. Follow [Understanding Dimension Reduction with Principal Component Analysis (PCA)](https://blog.paperspace.com/dimension-reduction-with-principal-component-analysis/) to do the same analysis using the DataFrame-based PCA `pca.fit()` from `pyspark.ml`.
2. Follow this lab to verify that using the other two RDD-based PCA APIs `computePrincipalComponents` and `computeSVD` will give the same PCA features.

## 4. Additional ideas to explore (*optional*)

### [HR analytics](https://rstudio-pubs-static.s3.amazonaws.com/345463_37f54d1c948b4cdfa181541841e0db8a.html)

A company is trying to figure out why their best and experienced employees are leaving prematurely from a [dataset](https://www.kaggle.com/liujiaqi/hr-comma-sepcsv). Follow the example [Principal Component Analysis explained on Kaggle](https://www.kaggle.com/nirajvermafcb/principal-component-analysis-explained) to perform such analysis in PySpark, using as many PySpark APIs as possible.


### Word meaning extraction

Use PySpark to perform the steps in IBM's notebook on [Spark-based machine learning for word meanings](https://github.com/IBMDataScience/word2vec/blob/master/Spark-based%20machine%20learning%20for%20word%20meanings.ipynb) that makes use of PCA, kmeans, and Word2Vec to learn word meanings.

### Bag of words analysis

Choose a [Bag of Words Data Set](https://archive.ics.uci.edu/ml/datasets/Bag+of+Words). Let us take  the **NIPS full papers** data as an example. 

The format of this data is

```markdown
    Number of documents
    Number of words in the vocabulary
    Total number of words in the collection
    docID wordID count
    docID wordID count
    ...
    docID wordID count
```

Our data matrix will be $doc \times wordcount$. To begin, we need to read this data in. Possible steps would include:

1. extract the number of documents and the size of the vocabulary, and strip off the first 3 lines
2. combine the words per document
3. create sparse vectors (for better space efficiency)

Start from a small dataset to test your work, and then checking **whether** your work scales up to the big **NYTIMES** bagofwords data. Keep everything as parallel as possible.

### Large image datasets

Find some large-scale image datasets to examine the principal components and explore low-dimensional representations.
