import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy import linalg

np.random.seed(44)
input_size = 3
#data = np.genfromtxt("lorenz_dataset.csv",delimiter = ',').T[0:1, :]
data = np.genfromtxt("lorenz_dataset.csv",delimiter = ',').T

print(data.shape)
res_size = 1000
a = 0.6 #leaking rate
#sparsity = 0.6
Win = (np.random.rand(res_size,1+input_size)-0.5)
W = (np.random.rand(res_size,res_size) - 0.5)
#mask = np.random.rand(res_size, res_size) >= sparsity
#W = W*mask
rho = max(abs(linalg.eig(W)[0]))
W *= 0.5 /rho
print(Win.shape)

train_len = 5000
test_len = 3000
init_len = 1000
X = np.zeros((res_size,train_len - init_len))
print(X.shape)
Yt = data[:,init_len+1:train_len+1]
print(Yt.shape)
x = np.zeros((res_size,1))
for t in range(train_len):
    u = data[:,t].reshape(-1, 1)
    x = (1-a)*x + a*np.tanh(np.dot(Win,np.vstack((u,0.1))) + np.dot(W,x))
    if t>=init_len:
        X[:,[t-init_len]] = x

#ridge_model = Ridge(alpha=1e-7)
#ridge_model.fit(X.T, Yt.T)
#Wout = ridge_model.coef_
#print(Wout.shape)

reg = 1e-7
Wout = linalg.solve( np.dot(X,X.T) + reg*np.eye(res_size),np.dot(X,Yt.T) ).T

Y = np.zeros((input_size,test_len))
u = data[:,train_len].reshape(-1, 1)
print(u[0])
print(u[1])
print(u[2])
#u[0] = u[0]+ 1e-9

for t in range(test_len):
    x = (1-a)*x + a*np.tanh(np.dot(Win,np.vstack((u,0.1))) + np.dot(W,x))
    y = np.dot(Wout,x)
    Y[:,[t]] = y
    u=y

errorLen = 360
sq_dist = np.sum((data[:,train_len+1:train_len+errorLen+1]-Y[:,0:errorLen])**2,axis = 0)
msd = np.mean(sq_dist)
rmse = np.sqrt(msd)
print('RMSE = ' + str( rmse ))
np.savetxt("ESN1.csv", Y.T, delimiter=",", fmt="%0.15f")
print(Y.shape)
print(Y[1])
print(Y[2])

fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
ax.plot(Y[0,:],Y[1,:],Y[2,:],lw = 0.5)
ax.plot(data[0,train_len:train_len+test_len],data[1,train_len:train_len+test_len],data[2,train_len:train_len+test_len],lw= 0.5,color = 'red')
plt.show()

plt.figure(1)
plt.plot(data[0,train_len+1:train_len+test_len+1], 'g' )
plt.plot(Y[0],'b')
plt.plot()

plt.figure(2)
plt.plot(data[1,train_len+1:train_len+test_len+1], 'g' )
plt.plot(Y[1],'b')
plt.plot()

plt.figure(3)
plt.plot(data[2,train_len+1:train_len+test_len+1], 'g' )
plt.plot(Y[2],'b')
plt.plot()
plt.show()