import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import pandas as pd

df=pd.read_csv("D:\Thesis\messy work\Final.csv")
df['TURBIDITY']=pd.to_numeric(df['TURBIDITY'],errors='coerce')
df['Labels']=df.Labels.astype(float)
df=df[['TEMPL','PHL','EC','CHLORIDE','TALKAL','TURBIDITY','DO','BOD','Labels']]

XRaw=np.array(df)
XRaw=np.random.permutation(XRaw)

def norm(X):
    meanAr=np.mean(X,axis=1)
    meanAr=np.reshape(meanAr,(meanAr.shape[0],1))
    varAr=np.var(X,axis=1)
    varAr=np.reshape(varAr,(varAr.shape[0],1))
    normX=np.divide(np.subtract(X,meanAr),varAr)
    return normX,meanAr,varAr

def normRet(X,XTrainMean,XTrainVar):
    normX=np.divide(np.subtract(X,XTrainMean),XTrainVar)
    return normX

Val=XRaw[0:400,:]
Test=XRaw[400:800,:]
Train=XRaw[800:-1,:]

def oneHot(labels,C):
    C=tf.constant(C)
    oneHotMat=tf.one_hot(labels,C,axis=0)
    sess=tf.Session()
    oneHot=sess.run(oneHotMat)
    sess.close()
    return oneHot

XTrainR=np.transpose(Train[:,0:-1])
XTrain,Xmean,Xvar=norm(XTrainR)
YTrain=Train[:,-1]
YTrain=np.transpose(YTrain)    #No need to reshape as one_hot handles the the missing dimension automatically
YTrain=oneHot(YTrain,10)

XTestR=np.transpose(Test[:,0:-1])
XTest=normRet(XTestR,Xmean,Xvar)
YTest=Test[:,-1]
YTest=np.transpose(YTest)
YTest=oneHot(YTest,10)

XValR=np.transpose(Val[:,0:-1])
XVal=normRet(XValR,Xmean,Xvar)
YVal=Val[:,-1]
YVal=np.transpose(YVal)
YVal=oneHot(YVal,10)


print(XTrainR,XTrainR.shape)
print(YTrain,YTrain.shape)
print(XTrain,XTrain.shape)

def createPlace(nX,nY):
    X=tf.placeholder(tf.float32,shape=(nX,None))
    Y=tf.placeholder(tf.float32,shape=(nY,None))
    return X,Y

def initParameters():
    W1 = tf.get_variable("W1",[30,8],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
    b1 = tf.get_variable("b1", [30,1], initializer = tf.zeros_initializer(),dtype=tf.float32)
    W2 = tf.get_variable("W2",[30,30],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
    b2 = tf.get_variable("b2", [30,1], initializer = tf.zeros_initializer(),dtype=tf.float32)
    W3 = tf.get_variable("W3",[30,30],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
    b3 = tf.get_variable("b3", [30,1], initializer = tf.zeros_initializer(),dtype=tf.float32)
    W4 = tf.get_variable("W4",[30,30],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
    b4 = tf.get_variable("b4", [30,1], initializer = tf.zeros_initializer(),dtype=tf.float32)
    W5 = tf.get_variable("W5",[20,30],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
    b5 = tf.get_variable("b5", [20,1], initializer = tf.zeros_initializer(),dtype=tf.float32)
    W6 = tf.get_variable("W6",[10,20],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
    b6 = tf.get_variable("b6", [10,1], initializer = tf.zeros_initializer(),dtype=tf.float32)
    
    
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
                  "b6": b6}

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

    return Z6

def compCost(Z6,Y):
    logits=tf.transpose(Z6)
    labels=tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
    return cost

def model(XTrainR,YTrain,XValR,YVal,learning_rate=.001,num_epochs=1000,print_cost=True):
    ops.reset_default_graph()
    (nX,m)=XTrainR.shape
    nY=YTrain.shape[0]
    costs=[]
    
    X,Y=createPlace(nX,nY)
    parameters=initParameters()
    Z6=forward(X,parameters)
    cost=compCost(Z6,Y)

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        #print(XTrainR)
        #print(YTrain)
        #print(XTrainR.shape)
        #print(YTrain.shape)
        for epoch in range(num_epochs):
            epoch_cost=0
            _,epoch_cost=sess.run([optimizer,cost],feed_dict={X:XTrainR,Y:YTrain})
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

        correct_prediction=tf.equal(tf.argmax(Z6),tf.argmax(Y))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:",accuracy.eval({X:XTrainR,Y:YTrain}))
        print ("Test Accuracy:", accuracy.eval({X:XValR,Y:YVal}))

        return parameters

parameters=model(XTrainR,YTrain,XValR,YVal,learning_rate=.001,num_epochs=50000,print_cost=True)