import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import pandas as pd
import math
#df=pd.read_csv("D:\Thesis\messy work\Final.csv")
#df['TURBIDITY']=pd.to_numeric(df['TURBIDITY'],errors='coerce')         #NEW
#df['Labels']=df.Labels.astype(float)                                   #NEW  
#df=df[['TEMPL','PHL','EC','CHLORIDE','TALKAL','TURBIDITY','DO','BOD','Labels']]
#print(df.dtypes)
#XRaw=np.array(df)
#XRaw=np.random.permutation(XRaw)
#pan=pd.DataFrame(XRaw)
#pan.to_csv("D:\Thesis\messy work\ReadyForDeploy.csv")

df=pd.read_csv("D:\Thesis\messy work\ReadyData.csv")
df=df[['0','1','2','3','4','5','6','7','8']]
#print(df.head())
#print(df.dtypes)
XRaw=np.array(df)
#print(XRaw.shape)

def norm(X):
    meanAr=np.mean(X,axis=1)
    meanAr=np.reshape(meanAr,(meanAr.shape[0],1))
    varAr=np.var(X,axis=1)
    varAr=np.reshape(varAr,(varAr.shape[0],1))
    stDev=np.sqrt(np.add(varAr,1e-8))
    normX=np.divide(np.subtract(X,meanAr),stDev)
    return normX,meanAr,stDev

def normRet(X,XTrainMean,XTrainStd):
    normX=np.divide(np.subtract(X,XTrainMean),XTrainStd)
    return normX

def oneHot(labels,C):
    C=tf.constant(C)
    oneHotMat=tf.one_hot(labels,C,axis=0)
    sess=tf.Session()
    oneHot=sess.run(oneHotMat)
    sess.close()
    return oneHot

Val=XRaw[0:100,:]
Test=XRaw[100:200,:]
Train=XRaw[200:-1,:]
#print(Train.shape)
XTrainR=np.transpose(Train[:,0:-1])
XTrain,Xmean,XStd=norm(XTrainR)
YTrain=Train[:,-1]
YTrain=np.transpose(YTrain)    #No need to reshape as one_hot handles the the missing dimension automatically
YTrain=oneHot(YTrain,10)

XTestR=np.transpose(Test[:,0:-1])
XTest=normRet(XTestR,Xmean,XStd)
YTest=Test[:,-1]
YTest=np.transpose(YTest)
YTest=oneHot(YTest,10)

XValR=np.transpose(Val[:,0:-1])
XVal=normRet(XValR,Xmean,XStd)
YVal=Val[:,-1]
YVal=np.transpose(YVal)
YVal=oneHot(YVal,10)


#print(XTrainR)
#XTurbi=XTrainR[-3,:]
#meanAr=np.mean(XTurbi,axis=1)
#meanAr=np.reshape(meanAr,(meanAr.shape[0],1))
#print(meanAr)

#print(XTurbinor,Xmean,Xvar)
print(XTrainR,XTrainR.shape)
print(YTrain,YTrain.shape)
print(XTrain,XTrain.shape)

def createPlace(nX,nY):
    X=tf.placeholder(tf.float32,shape=(nX,None))
    Y=tf.placeholder(tf.float32,shape=(nY,None))
    return X,Y

def initParameters():
    W1 = tf.get_variable("W1",[40,8],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
    b1 = tf.get_variable("b1",[40,1],initializer=tf.zeros_initializer(),dtype=tf.float32)
    W2 = tf.get_variable("W2",[50,40],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
    b2 = tf.get_variable("b2",[50,1],initializer=tf.zeros_initializer(),dtype=tf.float32)
    W3 = tf.get_variable("W3",[50,50],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
    b3 = tf.get_variable("b3",[50,1],initializer=tf.zeros_initializer(),dtype=tf.float32)
    W4 = tf.get_variable("W4",[50,50],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
    b4 = tf.get_variable("b4",[50,1],initializer=tf.zeros_initializer(),dtype=tf.float32)
    W5 = tf.get_variable("W5",[50,50],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
    b5 = tf.get_variable("b5",[50,1],initializer=tf.zeros_initializer(),dtype=tf.float32)
    W6 = tf.get_variable("W6",[50,50],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
    b6 = tf.get_variable("b6",[50,1],initializer=tf.zeros_initializer(),dtype=tf.float32)
    W7 = tf.get_variable("W7",[40,50],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
    b7 = tf.get_variable("b7",[40,1],initializer=tf.zeros_initializer(),dtype=tf.float32)
    W8 = tf.get_variable("W8",[35,40],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
    b8 = tf.get_variable("b8",[35,1],initializer=tf.zeros_initializer(),dtype=tf.float32)
    W9 = tf.get_variable("W9",[25,35],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
    b9 = tf.get_variable("b9",[25,1],initializer=tf.zeros_initializer(),dtype=tf.float32)
    W10 = tf.get_variable("W10",[10,25],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
    b10 = tf.get_variable("b10",[10,1],initializer=tf.zeros_initializer(),dtype=tf.float32)
    
    
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5,
                  "W6": W6,
                  "b6": b6,
                  "W7": W7,
                  "b7": b7,
                  "W8": W8,
                  "b8": b8,
                  "W9": W9,
                  "b9": b9,
                  "W10": W10,
                  "b10": b10,}
                  
    return parameters

def forward(X,parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    W5 = parameters['W5']
    b5 = parameters['b5']
    W6 = parameters['W6']
    b6 = parameters['b6']
    W7 = parameters['W7']
    b7 = parameters['b7']
    W8 = parameters['W8']
    b8 = parameters['b8']
    W9 = parameters['W9']
    b9 = parameters['b9']
    W10 = parameters['W10']
    b10 = parameters['b10']

    Z1 = tf.add(tf.matmul(W1,X),b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2),b3)
    A3 = tf.nn.relu(Z3)
    Z4 = tf.add(tf.matmul(W4,A3),b4)
    A4 = tf.nn.relu(Z4)
    Z5 = tf.add(tf.matmul(W5,A4),b5)
    A5 = tf.nn.relu(Z5)
    Z6 = tf.add(tf.matmul(W6,A5),b6)
    A6 = tf.nn.relu(Z6)
    Z7 = tf.add(tf.matmul(W7,A6),b7)
    A7 = tf.nn.relu(Z7)
    Z8 = tf.add(tf.matmul(W8,A7),b8)
    A8 = tf.nn.relu(Z8)
    Z9 = tf.add(tf.matmul(W9,A8),b9)
    A9 = tf.nn.relu(Z9)
    Z10 = tf.add(tf.matmul(W10,A9),b10)
    
    return Z10



def compCost(Z10,Y):
    logits=tf.transpose(Z10)
    labels=tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
    return cost

def random_mini_batches(X, Y, mini_batch_size = 64):
    m = X.shape[1]                  
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:,(k*mini_batch_size):((k+1)*mini_batch_size)]
        mini_batch_Y = shuffled_Y[:,(k*mini_batch_size):((k+1)*mini_batch_size)]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:,(num_complete_minibatches*mini_batch_size):m+1]
        mini_batch_Y = shuffled_Y[:,(num_complete_minibatches*mini_batch_size):m+1]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def model(XTrainR,YTrain,XValR,YVal,learning_rate=.001,minibatch_size=64,num_epochs=1000,print_cost=True):
    ops.reset_default_graph()
    (nX,m)=XTrainR.shape
    nY=YTrain.shape[0]
    costs=[]
    
    X,Y=createPlace(nX,nY)
    parameters=initParameters()
    Z10=forward(X,parameters)
    cost=compCost(Z10,Y)

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        #print(XTrainR)
        #print(YTrain)
        #print(XTrainR.shape)
        #print(YTrain.shape)
        for epoch in range(num_epochs+1):
            epoch_cost=0
            num_minibatches=int(m / minibatch_size)
            minibatches=random_mini_batches(XTrainR,YTrain,minibatch_size)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y)=minibatch
                
                _,minibatch_cost=sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y})
                epoch_cost+=minibatch_cost/num_minibatches
                #print(epoch_cost)
            if print_cost == True and epoch % 100 == 0:
                    print ("Cost after epoch %i: %f" % (epoch,epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                    costs.append(epoch_cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        correct_prediction=tf.equal(tf.argmax(Z10),tf.argmax(Y))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:",accuracy.eval({X:XTrainR,Y:YTrain}))
        print ("Test Accuracy:", accuracy.eval({X:XValR,Y:YVal}))

        return parameters

parameters=model(XTrain,YTrain,XVal,YVal,learning_rate=.0001,num_epochs=20000,print_cost=True)

parameters=model(XTrain,YTrain,XVal,YVal,learning_rate=.0001,num_epochs=20000,print_cost=True)

parameters=model(XTrain,YTrain,XVal,YVal,learning_rate=.0001,num_epochs=50000,print_cost=True)

parameters=model(XTrain,YTrain,XVal,YVal,learning_rate=.00005,minibatch_size=64,num_epochs=7000,print_cost=True)

parameters=model(XTrain,YTrain,XVal,YVal,learning_rate=.001,minibatch_size=64,num_epochs=1000,print_cost=True)
