import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def get_dataset(file_name):
    to_split = file_name.split("/")[-1]
    return to_split.split("_")[0]

file_name = sys.argv[1]
matplotlib.rcParams['figure.figsize'] = 1.11, .8
matplotlib.rcParams['font.size'] = 7
matplotlib.rcParams['font.sans-serif'] = "Helvetica"
matplotlib.rcParams['font.family'] = "sans-serif"

data = np.loadtxt(file_name, delimiter=',')
sorted_data = data[np.argsort(data[:,0])]
plt.figure()
for i in range(sorted_data.shape[1]-1):
    plt.plot(sorted_data[:,0],sorted_data[:,i+1])
plt.title(get_dataset(file_name))
plt.xlabel("Nt")
if "Nt" in file_name.split("/"):
    plt.ylabel("LBR")
elif "time" in file_name.split("/"):
    plt.ylabel("Time (ms)")
plt.show()

