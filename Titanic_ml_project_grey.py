#Loren Grey

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.preprocessing import StandardScaler

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
  return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
  a = x
  return (a > 0)*a

def relu_deriv(x):
  a = x
  return (a > 0)*1

def nnpred(x, theta1, theta2, b1, b2,thresh):
  h1t = x.T
  h2t = relu(np.dot(theta1.T, h1t) + b1)
  h3t = sigmoid(np.dot(theta2.T, h2t) + b2)
  preds = (h3t>1/2)*1 # predicted labels
  if thresh==True:
    return preds
  else:
    return h3t

def myplot(x,y,phrase,boundary):
  plt.close('all')
  arg_0 = np.where(y == 0)
  arg_1 = np.where(y == 1)
  arg_2 = np.where(y == 2)

  fig, ax = plt.subplots()
  if boundary == True:
    ax.pcolormesh(xx, yy, Z, cmap=plt.cm.Set1)  # plot in plane based on values of Z. Note: Z is binary at locations
    ax.contour(xx, yy, Z, [0.5], colors='w', linewidths=2)

  passed = ax.scatter(x[arg_1, 0], x[arg_1, 1], marker='o', s=60, color='c', label='first class')
  failed = ax.scatter(x[arg_0, 0], x[arg_0, 1], marker='d', s=60, color='b', label='third class')
  dropped = ax.scatter(x[arg_2, 0], x[arg_2, 1], marker='s', s=60, color='g', label='second class')
  ax.set_xlabel('Age');
  ax.set_ylabel('Fare')
  ax.set_title(f"{phrase}")
  ax.legend([passed, failed, dropped], ('1st Class', '2nd Class', '3rd Class'), loc=2)
  fig.show()
  plt.show()
  ax.set_xlim(x_min,x_max); ax.set_ylim(y_min,y_max);


# convert labels of single value to one-hot encoding
def to_one_hot(Y):
  binzd_len = int(np.amax(Y)+1)
  Z = np.zeros((binzd_len, len(Y)))
  for i in range(len(Y)):
    Z[int(Y[i]), i] = 1
  return Z

# convert one-hot encoding back to single value labels
def from_one_hot(Z):
  Y = np.zeros(Z.shape[1])
  for i in range(len(Y)):
    col = Z[:,i]
    for j in range(len(col)):
      if(col[j] == 1):
        Y[i] = j
  return Y


import numpy as np
L = []
with open('titanic_data.txt') as f:
    for line in f:
        L.append(line.split())

L2 = []
for k in L:
    L2.append(list(map(lambda x: float(x), k)))

data = np.array(L2)
x = data[:,0:2]
y_array = data[:,2]

y = to_one_hot(y_array).T

# # # #rescale the data
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# split the data into 75% training and 25% testing
x_train, x_test, y_train, y_test = train_test_split(x, y)
y_train, y_test = y_train.T, y_test.T
y_train_labels = from_one_hot(y_train)
y_test_labels = from_one_hot(y_test)

# for plotting
del_ax=0.75 # plotting boundary
x_min, x_max = x_train[:, 0].min(), x_train[:, 0].max()
y_min, y_max = x_train[:, 1].min(), x_train[:, 1].max()
x_min, x_max = x_min - del_ax * np.abs(x_min), x_max + del_ax * np.abs(x_max)
y_min, y_max = y_min - del_ax * np.abs(y_min), y_max + del_ax * np.abs(y_max)


myplot(x_train,y_train_labels,phrase='Titanic Passenger Class Nuero Net',boundary=False)

# weights
np.random.seed(6)
theta1 = 2*np.random.random((2, 5)) - 1 # layer 2. connects 2 neurons to 3
theta2 = 2*np.random.random((5, 3)) - 1 # layer 3. connects 3 neurons to 1
b1 = 2*np.random.random((5, 1)) - 1 # bias
b2 = 2*np.random.random((3, 1)) - 1 # bias


alpha = 0.01 #learning rate
N=1000 #number of epochs
numberNeurons = (1,2,3,1) #neurons per layer
mbsize = 75 #mini-batch size


loss = []
ind = [i for i in range(x_train.shape[0])]
mbsize = 75 # mini-batch size
npts = x_train.shape[0] # total # of training examples

for i in range(1,N+1):

  np.random.shuffle(ind)
  for j in range(0, int(npts / mbsize)):  # mini batch partition range

    print(f'epoch #{i:<5} using batch #{j + 1:<3} with data points from: {j * mbsize+1:>3} to {(j + 1) * mbsize: >3}')
    x_mb = x_train[ind[j*mbsize:(j + 1)*mbsize]] # current mini-batch
    print(x_mb.shape)
    if mbsize == 1:
      x_mb = np.array(x_mb)

    y_mb = y_train[:,ind[j*mbsize:(j + 1)*mbsize]]

    # feed forward pass
    h1 = x_mb.T
    h2 = relu(np.dot(theta1.T, h1) + b1)
    h3 = sigmoid(np.dot(theta2.T, h2) + b2)
    h_last = np.dot(theta2.T, h2) + b2

    # for output layer
    delta3 = 2*(h3-y_mb)*sigmoid_deriv(h_last)

    # backward pass
    delta2 = np.dot(theta2,delta3)*relu_deriv(np.dot(theta1.T,h1)+b1)

    grad_theta1 = np.dot(h1,delta2.T)
    grad_theta2 = np.dot(h2,delta3.T)
    grad_b1 = delta2
    grad_b2 = delta3

    theta1 = theta1 -alpha*grad_theta1
    theta2 = theta2 -alpha*grad_theta2
    b1 = b1 - alpha*np.sum(grad_b1)
    b2 = b2 - alpha*np.sum(grad_b2)


  if i%1== 0:
    h3p = nnpred(x_train, theta1, theta2, b1, b2, thresh=False)
    loss_current = np.mean((h3p-y_train)**2)
    loss.append(loss_current)
    pred = (h3p > 1 / 2) * 1
    accuracy = (1-np.mean(abs(pred-y_train)))*100
    #print(f"pred = {pred[0]}")
    #print(f"actu = {y}")
    print(50*'*')
    print(f"epoch = {i}'")
    print(f"current loss = {loss_current}")
    print(f"training accuracy = {accuracy}%")
    print(50 * '*')


# Plot the accuracy chart
plt.plot(loss)
plt.xlabel('Training Epoch')
plt.ylabel('Error')
plt.show()

# training accuracy
#print(f"Training Accuracy {round(accuracy, 2)}%")
preds_train = nnpred(x_train,theta1,theta2,b1,b2,thresh=True)
train_acc = (preds_train == y_train).mean()
print(f'Neural Net Training accuracy = {train_acc*100}%')

# testing accuracy
preds_test = nnpred(x_test,theta1,theta2,b1,b2,thresh=True)
test_acc = (preds_test == y_test).mean()
print(f'Neural Net Testing accuracy = {test_acc*100}%')



h = .01  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))  # create a mesh of pts

# x and y coords into a long vector and apply prediction to it
xxv = xx.ravel(); yyv = yy.ravel()
xv = np.array([xxv,yyv]).T
Z1 = nnpred(xv,theta1,theta2,b1,b2,thresh=True)
Z2 = from_one_hot(Z1)
# # Put the result into a color plot
Z = Z2.reshape(xx.shape)  #reshape it back into a plane so get predicted value at each mesh location


#plot training and decision boundary
myplot(x_train,y_train_labels,phrase='Neural Net: Training',boundary=True)

#plot testing and decision boundary
myplot(x_test,y_test_labels,phrase='Neural Net: Testing',boundary=True)





