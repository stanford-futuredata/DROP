# DROP
DROP is a system that efficiently performs dimensionality reduction via principal component analysis (PCA) using end-to-end workload optimization. You can refer to [our paper](https://arxiv.org/abs/1708.00183) for information on how DROP operates.

### Usage
We have provided a DROP demo script (`scripts/DROPDemo.py`) and sample dataset from the [UCR Time Series Repository](http://www.cs.ucr.edu/~eamonn/time_series_data/) (`data/labeled/wafer.csv`). 

From the project root directory, run `./scripts/DROPDemo.py` to build and run the demo. The transformed dataset will be written to `data/transformed`.
