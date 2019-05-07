import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

hist = np.load('histcomp1.npy')
hist2 = np.load('histcomp2.npy')
hist3 = np.load('histcomp3.npy')
hist4 = np.load('histcomp4.npy')
hist5 = np.load('histcomp5.npy')
hist6 = np.load('histcomp7.npy')
grav = np.load('grav.npy')
histgrav = np.load('histgrav.npy')
grav1 = np.load('grav1.npy')

# plt.figure()
# plt.scatter([x for x in range(len(hist))], hist)
# plt.show()

# plt.figure()
# plt.xlabel('Epochs')
# plt.ylabel('Score')
# plt.scatter([x for x in range(len(hist4))], hist4)
# plt.show()

# plt.figure()
# plt.scatter([x for x in range(len(hist9))], hist9)
# plt.scatter([x for x in range(len(hist10))], hist10)
# plt.show()

# print('mean9', np.mean(hist9))
# print('mean10', np.mean(hist10))

from mlxtend.plotting import category_scatter

# vals = [x for x in range(len(histgrav))]

# grav = grav.tolist()
# histgrav = histgrav.tolist()

# data = pd.DataFrame(np.array([vals,grav,histgrav]).T)
# data = data.rename(columns={0: "a", 1: "b", 2:'c'})

# print(data)

# plt.figure()
# plt.scatter(data['a'][data['b']==1], data['c'][data['b']==1], label='Low Grav')
# plt.scatter(data['a'][data['b']==4], data['c'][data['b']==4], label='High Grav')
# plt.legend()
# plt.xlabel('Epoch')
# plt.ylabel('Score')
# plt.show()


print(hist6)


print(np.mean(hist))
print(np.mean(hist2))
print(np.mean(hist3))
print(np.mean(hist5))
print(np.mean(hist6))
