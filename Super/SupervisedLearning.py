
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_rcv1
rcv1 = fetch_rcv1()


# In[2]:


# QUESTION 1

import numpy as np

# LABEL VECTOR
ccat_index = np.nonzero(rcv1['target'][:,33])[0]
label_vector = np.full((804414,1),-1)
label_vector[ccat_index] = 1
# TRAIN AND TEST VECTOR
train_vector_labels = label_vector[0:100000,:]
train_vector_features = rcv1['data'][0:100000,:]
test_vector_labels = label_vector[100000:,:]
test_vector_features = rcv1['data'][100000:,:]


# In[3]:


print("Q1 Done")


# In[76]:


# QUESTION 2
# Lec 7 - 11
# Inputs S, lambda, T, k
import random as r
import math
train_err10 = np.zeros(10000)
train_err1 = np.zeros(10000)
train_errd1 = np.zeros(10000)
train_err01 = np.zeros(10000)
train_err001 = np.zeros(10000)


def SVM_acc(w, X, Y):
    w1 = w.reshape(w.shape[0],1)
#     print(w1)
    Y_new = X*w1
#     print(Y_new)
    Y_new[Y_new <= 0] = -1
    Y_new[Y_new > 0] = 1
#     print(Y_new)
#     print(Y_new)
    result = Y-Y_new
    
    return 1. - (np.count_nonzero(result)/result.shape[0])

def generate_atplus(X, Y, w):
    w1 = w.reshape(w.shape[0],1)
#     print(w1)
    dot_prod = X*w1
#     print(dot_prod)
    final_dot = np.multiply(dot_prod,Y)
#     print(final_dot)
    final_index = np.where(final_dot < 1)
#     print(final_index[0])
    return final_index[0]
    

def pegasos(X, Y, l, T, k):
    wt = np.zeros((X.shape[1]))    
    for i in range(T):
        t = i+1
#         GENERATE MINIBATCH
        nt = 1./(t*l)
#         print(nt)
        At = np.array(sorted(r.sample(range(0,X.shape[0]),k)))
#         print(At)
        Atplus = generate_atplus(X[At,:],Y[At], wt)
#         print(Atplus)
        X1 = X[At[Atplus],:]
        Y1 = Y[At[Atplus]]
#         print(Y1)
#         print(X1)


# # #         mini-batch gradient SGD
        wtprime = (1-(nt*l))*wt
#         print(wtprime)
#         print(np.multiply(X1,Y1)[0])
        deltat = (nt/k)*(((X1.T)*Y1)).T[0]
#         print(deltat)
#         print(wtprime)
        wtprime = wtprime + deltat
#         print(wtprime)
# # #         deltat = (l*wt)- ((1/k) * np.sum( Y1.T * X1))
# # #         wtprime = wt - (nt*deltat)

        
# # #         Projection
        numer = 1/float(math.sqrt(l))
#         print(numer)
        denom = np.linalg.norm(wtprime,2)
#         print(denom)
        wt = min(1, numer/denom) * wtprime
#    Store     
#         w1 = wt.reshape(wt.shape[0],1)
#         Y_new = X*w1
#         Y_new[Y_new <= 0] = -1
#         Y_new[Y_new > 0] = 1
#         result = Y-Y_new
#         err = 1. - (np.count_nonzero(result)/result.shape[0])
#         if l == 10:
#             train_err10[i] = err
        
#         elif l == 1:
#             train_err1[i] = err
        
#         elif l ==.1:
#             train_errd1[i] = err
        
#         elif l ==.01:
#             train_err01[i] = err
#         else:
#             train_err001[i] = err
            
        
        
        
#         print(wt)
#     print(wt)
    return wt


# w = pegasos(train_vector_features,train_vector_labels, .001,10000,100)
# # print("3")
# print(SVM_acc(w,train_vector_features, train_vector_labels))
# print(SVM_acc(w,test_vector_features, test_vector_labels))

# w = pegasos(train_vector_features,train_vector_labels, .000001,10000,100)
# # print("3")
# print(SVM_acc(w,train_vector_features, train_vector_labels))
# print(SVM_acc(w,test_vector_features, test_vector_labels))


# pegasos(train_vector_features,train_vector_labels, .001,10000,100)
# print("5")

# print(SVM_acc(w,test_vector_features, test_vector_labels))


# In[38]:


# import matplotlib.pyplot as plt
# a = 1- train_err10[:10000]
# b = 1- train_err1[:10000]
# c = 1- train_errd1[:10000]
# d = 1- train_err01[:10000]
# e = 1- train_err001[:10000]
# # t = np.arange(0.,10000, 1000)
# # plt.ylim(0, 1., .1)
# plt.plot(a, 'r', label="lambda = 10")
# plt.plot(b, 'b', label="lambda = 1")
# plt.plot(c, 'g', label="lambda = .1")
# plt.plot(d, 'k', label="lambda = .01")
# plt.plot(e, 'm', label="lambda = .001")
# plt.title("PEGASOS Training Error vs Iteration for K = 100")
# plt.xlabel("Iteration")
# plt.ylabel("Training Error")
# plt.legend(loc='upper right')

# # plt.show()
# plt.savefig("PEGASOS_K100.png")


# In[78]:


# QUESTION 3
# Lec 11-13
import random as r
import math
atrain_err10 = np.zeros(10000)
atrain_err1 = np.zeros(10000)
atrain_errd1 = np.zeros(10000)
atrain_err01 = np.zeros(10000)
atrain_err001 = np.zeros(10000)

def adagrad_acc(w, X, Y):
    w1 = w.reshape(w.shape[0],1)
#     print(w1)
    Y_new = X*w1
#     print(Y_new)
    Y_new[Y_new <= 0] = -1
    Y_new[Y_new > 0] = 1
#     print(Y_new)
#     print(Y_new)
    result = Y-Y_new
    
    return 1. - (np.count_nonzero(result)/result.shape[0])

def generate_atplus(X, Y, w):
    
    w1 = w.reshape(w.shape[0],1)
#     print(w1)
    dot_prod = X*w1
#     print(dot_prod)
    final_dot = np.multiply(dot_prod,Y)
#     print(final_dot)
    final_index = np.where(final_dot < 1)
#     print(final_index[0])
    return final_index[0]
    

def adagrad(X, Y, l,T, k, nonzero_num):
    wt = np.zeros((X.shape[1]))
    gt = np.zeros((X.shape[1]))

    
    for i in range(T):
        t = i+1
        lr =  1./(t*l)


  #       TODO  GENERATE MINIBATCH/SGD
      
        #         print(t)
#         print(nt)
        At = np.array(sorted(r.sample(range(0,X.shape[0]),k)))
        Atplus = generate_atplus(X[At,:],Y[At], wt)
        
        
        X1 = X[At[Atplus],:]
        Y1 = Y[At[Atplus]]
        
        # # #         mini-batch gradient SGD
#         print(wtprime)
#         print(np.multiply(X1,Y1)[0])
        gti =  (l*wt) - ((1/k)*(((X1.T)*Y1)).T[0])
        
#         print(X1)
#         print(Y1)
#         print(gti)
#         print(gt)
#         print(wt)
        
#         ADAGRAD PART
#         print("ADAGRAD")
#         print(np.power(gti,2))
#         print(gt)
        squared_gti = gt + np.power(gti,2)
#         print(squared_gti)
        gt = squared_gti
#         print("ADAGRAD2")
#         print(gt)
        scaled_lr = (lr/((nonzero_num + np.sqrt(squared_gti))))
#         print(scaled_lr)
        wtprime = wt - (scaled_lr*gti)
#         print(wtprime)
#         print(gt)
                
#         #         Projection
#         mahalanobis =  np.linalg.norm(np.sqrt(squared_gti)*(wt-wtprime),2)
#         print(mahalanobis)
#         wt = np.argmin(mahalanobis)
        
        denom = np.linalg.norm((np.sqrt(squared_gti)*wtprime),2)
#         print(denom)
        numer = 1./math.sqrt(l)
        wt = min(1, numer/denom) * wtprime
#         print(wt)

#         w1 = wt.reshape(wt.shape[0],1)
#     #     print(w1)
#         Y_new = X*w1
#     #     print(Y_new)
#         Y_new[Y_new <= 0] = -1
#         Y_new[Y_new > 0] = 1
#     #     print(Y_new)
#     #     print(Y_new)
#         result = Y-Y_new
#         err = 1. - (np.count_nonzero(result)/result.shape[0])

#         if l == 10:
#             atrain_err10[i] = err
#         elif l == 1:
#             atrain_err1[i] = err
#         elif l ==.1:
#             atrain_errd1[i] = err
#         elif l ==.01:
#             atrain_err01[i] = err
#         else:
#             atrain_err001[i] = err
        
        
    return wt



# print("start")
# w = adagrad(train_vector_features,train_vector_labels, .001,5000,10, 1e-8)
# print(adagrad_acc(w,train_vector_features, train_vector_labels))
# print(adagrad_acc(w,test_vector_features, test_vector_labels))



# w = wt = adagrad(train_vector_features,train_vector_labels, .000001,5000,10, 1e-8)
# print("5")
# print(adagrad_acc(w,train_vector_features, train_vector_labels))
# print(adagrad_acc(w,test_vector_features, test_vector_labels))


# In[66]:


# import matplotlib.pyplot as plt
# a = 1- atrain_err10[:5000]
# b = 1- atrain_err1[:5000]
# c = 1- atrain_errd1[:5000]
# d = 1- atrain_err01[:5000]
# e = 1- atrain_err001[:5000]
# # t = np.arange(0.,10000, 1000)
# # plt.ylim(0, 1., .1)
# plt.plot(a, 'r', label="lambda = 10")
# plt.plot(b, 'b', label="lambda = 1")
# plt.plot(c, 'g', label="lambda = .1")
# plt.plot(d, 'k', label="lambda = .01")
# plt.plot(e, 'm', label="lambda = .001")
# plt.title("ADAGRAD Training Error vs Iteration for K = 100")
# plt.xlabel("Iteration")
# plt.ylabel("Training Error")
# plt.legend(loc='upper right')

# # plt.show()
# plt.savefig("ADAGRAD_K100.png")


# In[5]:


# QUESTION 4
#Fix vectors
label_vector = np.full((804414,1),0)
label_vector[ccat_index] = 1
train_vector_labels = label_vector[0:100000,:]
train_vector_features = rcv1['data'][0:100000,:]
test_vector_labels = label_vector[100000:,:]
test_vector_features = rcv1['data'][100000:,:]


# In[12]:


# QUESTION 4A.1
# NN and Keras - Lec 15 - 18

# optimized should be sgd
#  5 epochs
#  1 hidden layers with 100 hidden units

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
# input
model.add(Dense(units=100, activation= 'relu', input_dim=47236))
# model.add(Dense(units=100, activation= 'relu', input_dim=47236))
model.add(Dense(1, activation='relu'))
# # hidden
# model.add(Dense(units=100, activation= 'sigmoid'))
# # output
# model.add(Dense(units=100, activation= 'sigmoid'))



model.compile(loss ='mean_squared_error', optimizer='sgd', metrics=['acc'])
# binary_crossentropy
print("MODEL DONE")

model.fit(train_vector_features, train_vector_labels, epochs=5, batch_size=100)
print("FIT DONE")
score = model.evaluate(train_vector_features, train_vector_labels)

print(score)
# optimized should be sgd
#  X epochs
#  at most 6 hidden layers with 100 hidden units


# In[13]:


# QUESTION 4A.2
# NN and Keras - Lec 15 - 18

# optimized should be sgd
#  5 epochs
#  1/2/3 hidden layers with 100 hidden units
model = Sequential()
# input
model.add(Dense(units=100, activation= 'relu', input_dim=47236))
model.add(Dense(units=100, activation= 'relu'))
model.add(Dense(1, activation='relu'))
# # hidden
# model.add(Dense(units=100, activation= 'sigmoid'))
# # output
# model.add(Dense(units=100, activation= 'sigmoid'))



model.compile(loss ='mean_squared_error', optimizer='sgd', metrics=['acc'])

print("MODEL DONE")

model.fit(train_vector_features, train_vector_labels, epochs=5, batch_size=100)
print("FIT DONE")
score = model.evaluate(train_vector_features, train_vector_labels)
print(score)


# In[14]:


# QUESTION 4A.2
# NN and Keras - Lec 15 - 18

# optimized should be sgd
#  5 epochs
#  1/2/3 hidden layers with 100 hidden units
model = Sequential()
# input
model.add(Dense(units=100, activation= 'relu', input_dim=47236))
model.add(Dense(units=100, activation= 'relu'))
model.add(Dense(units=100, activation= 'relu'))
model.add(Dense(1, activation='relu'))
# # hidden
# model.add(Dense(units=100, activation= 'sigmoid'))
# # output
# model.add(Dense(units=100, activation= 'sigmoid'))



model.compile(loss ='mean_squared_error', optimizer='sgd', metrics=['acc'])

print("MODEL DONE")

model.fit(train_vector_features, train_vector_labels, epochs=5, batch_size=100)
print("FIT DONE")
score = model.evaluate(train_vector_features, train_vector_labels)
print(score)


# In[28]:


import matplotlib.pyplot as plt
y = np.array([1,2,3,4,5])
a= np.array([.6210, .8424, .8902, .9043, .9120])
b= np.array([.6141, .8633, .9084, .9214, .9294])
c= np.array([.6087, .8835, .9205, .9320, .9384])
plt.plot(y, a, 'r', label="1 hidden layer")
plt.plot(y,b, 'b', label="2 hidden layers")
plt.plot(y, c, 'g', label="3 hidden layers")
plt.xticks(np.arange(1, 6, step=1))
# plt.plot(d, 'k', label="lambda = .01")
# plt.plot(e, 'm', label="lambda = .001")
# plt.title("ADAGRAD Training Error vs Iteration for K = 100")
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.legend(loc='lower right')

# plt.show()
plt.savefig("NNAcc_K100.png")
plt.show()
y = np.array([1,2,3])
a= 1- np.array([.91367, .93121, .94233])
plt.plot(y, a, 'r')
plt.xticks(np.arange(1, 4, step=1))
plt.xlabel("# of Hidden Layers")
plt.ylabel("Training Error")

# plt.show()
plt.savefig("NNTE_K100.png")


# In[37]:


model = Sequential()
# input
model.add(Dense(units=50, activation= 'relu', input_dim=47236))
model.add(Dense(1, activation='relu'))
# # hidden
# model.add(Dense(units=100, activation= 'sigmoid'))
# # output
# model.add(Dense(units=100, activation= 'sigmoid'))

# a= 1- np.array([.91367, .93121, .94233])
# a= np.array([.6210, .8424, .8902, .9043, .9120])
# b= np.array([.6141, .8633, .9084, .9214, .9294])
# c= np.array([.6087, .8835, .9205, .9320, .9384])

model.compile(loss ='mean_squared_error', optimizer='sgd', metrics=['acc'])
# binary_crossentropy
print("MODEL DONE")

model.fit(train_vector_features, train_vector_labels, epochs=5, batch_size=100)
print("FIT DONE")
score = model.evaluate(train_vector_features, train_vector_labels)

print(score)
# optimized should be sgd
#  X epochs
#  at most 6 hidden layers with 100 hidden units


# In[38]:


model = Sequential()
# input
model.add(Dense(units=200, activation= 'relu', input_dim=47236))
model.add(Dense(1, activation='relu'))
# # hidden
# model.add(Dense(units=100, activation= 'sigmoid'))
# # output
# model.add(Dense(units=100, activation= 'sigmoid'))

# a= 1- np.array([.91367, .93121, .94233])
# a= np.array([.6210, .8424, .8902, .9043, .9120])
# b= np.array([.6141, .8633, .9084, .9214, .9294])
# c= np.array([.6087, .8835, .9205, .9320, .9384])

model.compile(loss ='mean_squared_error', optimizer='sgd', metrics=['acc'])
# binary_crossentropy
print("MODEL DONE")

model.fit(train_vector_features, train_vector_labels, epochs=5, batch_size=100)
print("FIT DONE")
score = model.evaluate(train_vector_features, train_vector_labels)

print(score)


# In[41]:


model = Sequential()
# input
model.add(Dense(units=50, activation= 'relu', input_dim=47236))
model.add(Dense(units=50, activation= 'relu'))
model.add(Dense(1, activation='relu'))
# # hidden
# model.add(Dense(units=100, activation= 'sigmoid'))
# # output
# model.add(Dense(units=100, activation= 'sigmoid'))



model.compile(loss ='mean_squared_error', optimizer='sgd', metrics=['acc'])
# binary_crossentropy
print("MODEL DONE")

model.fit(train_vector_features, train_vector_labels, epochs=5, batch_size=100)
print("FIT DONE")
score = model.evaluate(train_vector_features, train_vector_labels)

print(score)


# In[45]:


model = Sequential()
# input
model.add(Dense(units=200, activation= 'relu', input_dim=47236))
model.add(Dense(units=200, activation= 'relu'))
model.add(Dense(1, activation='relu'))
# # hidden
# model.add(Dense(units=100, activation= 'sigmoid'))
# # output
# model.add(Dense(units=100, activation= 'sigmoid'))



model.compile(loss ='mean_squared_error', optimizer='sgd', metrics=['acc'])
# binary_crossentropy
print("MODEL DONE")

model.fit(train_vector_features, train_vector_labels, epochs=3, batch_size=100)
print("FIT DONE")
score = model.evaluate(train_vector_features, train_vector_labels)

print(score)


# In[44]:


model = Sequential()
# input
model.add(Dense(units=50, activation= 'relu', input_dim=47236))
model.add(Dense(units=50, activation= 'relu'))
model.add(Dense(units=50, activation= 'relu'))
model.add(Dense(units=50, activation= 'relu'))
model.add(Dense(units=50, activation= 'relu'))
model.add(Dense(units=50, activation= 'relu'))
model.add(Dense(1, activation='relu'))
# # hidden
# model.add(Dense(units=100, activation= 'sigmoid'))
# # output
# model.add(Dense(units=100, activation= 'sigmoid'))



model.compile(loss ='mean_squared_error', optimizer='sgd', metrics=['acc'])
# binary_crossentropy
print("MODEL DONE")

model.fit(train_vector_features, train_vector_labels, epochs=5, batch_size=100)
print("FIT DONE")
score = model.evaluate(train_vector_features, train_vector_labels)

print(score)


# In[46]:


model = Sequential()
# input
model.add(Dense(units=100, activation= 'relu', input_dim=47236))
model.add(Dense(units=100, activation= 'relu'))
model.add(Dense(units=100, activation= 'relu'))
model.add(Dense(units=100, activation= 'relu'))
model.add(Dense(units=100, activation= 'relu'))
model.add(Dense(units=100, activation= 'relu'))
model.add(Dense(1, activation='relu'))
# # hidden
# model.add(Dense(units=100, activation= 'sigmoid'))
# # output
# model.add(Dense(units=100, activation= 'sigmoid'))



model.compile(loss ='mean_squared_error', optimizer='sgd', metrics=['acc'])

print("MODEL DONE")

model.fit(train_vector_features, train_vector_labels, epochs=5, batch_size=100)
print("FIT DONE")
score = model.evaluate(train_vector_features, train_vector_labels)
print(score)


# In[50]:


model = Sequential()
# input
model.add(Dense(units=50, activation= 'relu', input_dim=47236))
model.add(Dense(units=50, activation= 'relu'))
model.add(Dense(units=50, activation= 'relu'))
model.add(Dense(units=50, activation= 'relu'))
model.add(Dense(units=50, activation= 'relu'))
model.add(Dense(1, activation='relu'))
# # hidden
# model.add(Dense(units=100, activation= 'sigmoid'))
# # output
# model.add(Dense(units=100, activation= 'sigmoid'))



model.compile(loss ='mean_squared_error', optimizer='sgd', metrics=['acc'])

print("MODEL DONE")

model.fit(train_vector_features, train_vector_labels, epochs=5, batch_size=50)
print("FIT DONE")
score = model.evaluate(train_vector_features, train_vector_labels)
print(score)


# In[52]:


model = Sequential()
# input
model.add(Dense(units=100, activation= 'relu', input_dim=47236))
model.add(Dense(units=100, activation= 'relu'))
model.add(Dense(units=100, activation= 'relu'))
model.add(Dense(units=100, activation= 'relu'))
model.add(Dense(units=100, activation= 'relu'))
model.add(Dense(units=100, activation= 'relu'))
model.add(Dense(1, activation='relu'))
# # hidden
# model.add(Dense(units=100, activation= 'sigmoid'))
# # output
# model.add(Dense(units=100, activation= 'sigmoid'))



model.compile(loss ='mean_squared_error', optimizer='sgd', metrics=['acc'])

print("MODEL DONE")

model.fit(train_vector_features, train_vector_labels, epochs=5, batch_size=100)
print("FIT DONE")
score = model.evaluate(test_vector_features, test_vector_labels)
print(score)

