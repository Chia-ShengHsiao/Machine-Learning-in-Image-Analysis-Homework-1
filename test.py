import numpy as np
import random
from datetime import datetime
from tabulate import tabulate

#**************************************************problem 1************************************************
def Problem1(N, D):
    print('I use ramdon to create a matrix X for Problem1, X as following :')
    X = np.random.randint(0,100, (N,D))
    #use random to create a matrix X
    print X
    #show the matrix
    Z=compute_distance_native(X)
    #use two for loop to get Z
    print('')
    print('Problem 1, use two loop to get Z. After computing, the Z is :')
    print Z
    
    Z=compute_distance_smart(X)
    #without using loop to get Z
    print('')
    print('Problem 1,without using loop to get Z. After computing, the Z is :')
    print Z 
    return

#*************************solution 1********************

def compute(temp1, temp2, D):
    #compute xi-xj, return a value
    temp = 0
    for i in range(0,D,1):
        temp = temp + (temp1[0,i] - temp2[0,i])*(temp1[0,i] - temp2[0,i])
        #(xi-xj)*(xi-xj)T
    temp = temp**0.5
    #Square root the value
    return temp

def compute_distance_native(X):
    N=X.shape[0]
    #num of row
    D=X[0].shape[0]
    #num of column
    
    #use two for loop to get Z & print Z
    Z = np.zeros((N,N), dtype=np.int64)
    #initial a new NumPy for Z
    for i in range(0,N,1):
        for j in range(0,N,1):
            temp1 = X[i :]
            #first vector xi
            temp2 = X[j :]
            #second vector xj
            Z[i,j]=compute(temp1, temp2, D)
            #save into Z
    return Z
#*************************solution 2********************
def compute_distance_smart(X):
    N=X.shape[0]
    #num of row
    D=X[0].shape[0]
    #num of column
    
    #get Z & print Z without any loop
    XT = X.T
    M = X.dot(XT)
    D = np.diag(np.diag(M))
    Xi=np.zeros((N,N), dtype=np.int64)
    #create a Xi numpy, the type is int64, all values are 0
    Xi.fill(1)
    #all index of Xi change value from 0 to 1 
    Xi=D.dot(Xi)
    Xj = Xi.T
    
    Z = Xi-2*M+Xj
    #Xi^2-2XiT*Xj+Xj^2
    Z=np.sqrt(Z)
    #Square root the value
    Z=np.int64(Z)
    #use int to remove decimal point
    return Z
  
#**************************************************problem 2************************************************
def Problem2(N,D):
    print('I use ramdon to create a matrix X for Problem2, X as following :')
    X = np.random.randint(0,100, (N_p2,D_p2))
    print X
    #use random to create a matrix X
    R=compute_correlation_native(X)
    #use two for loop to get R
    print('') 
    print('Problem 2, use loop to get R. After computing, the R is :')
    R=np.round(R,2)  
    #The value will be between 0 and 1; therefore, I 
    print R
      
    R=compute_correlation_smart(X)
    #without using loop to get R
    print('') 
    print('Problem 2, without using loop to get R. After computing, the R is :')
    print R
    return


#*************************solution 1********************
def compute_correlation_native(X):
    N=X.shape[0]
    #num of row
    D=X[0].shape[0]
    #num of column
    
    
    #get R with using loop
    S=np.zeros((D,D), dtype=np.int64)
    R=np.zeros([D,D])
    # R may be smaller than 1 and not a integer; therefore I don't decide dtype int64, and the default is float
    meanR = X.mean(0)
    #get the average vector of row 
    meanC = X.mean(1)
    #get the average vector of column
    
    for i in range(0, D, 1):
    #compute S
        for j in range(0, D, 1):
            temp=0
            for k in range(0, N, 1):
                temp = temp + (X[k, i] -meanR[i])*(X[k, j]-meanC[j])
                #compute (Xn,i - ui)(Xn,j-uj) from 0 to N-1
            S[i,j] = (temp / (N-1)) 
            #divide N-1
    
    for i in range(0, D, 1):
    #compute R
        for j in range(0, D, 1):
            if i==j :
                #the value of slash will be 1
                R[i,j]=1
            else:
                #compute R[i,j], where i is not equal to j
                R[i,j]= S[i,j] /( (S[i,i]**0.5)*(S[j,j]**0.5) )          
    return R

#*************************solution 2********************

def compute_correlation_smart(X):
    N=X.shape[0]
    #num of row
    D=X[0].shape[0]
    #num of column
    
    #get R without using loop
    temp = np.zeros((N,N))
    #create a new N*N numpy
    temp.fill(1)
    #all index fill 1
    
    TD = temp.dot(X)/N
    temp2 = X-TD
    temp2T = temp2.T
    
    S=temp2T.dot(temp2) / (N-1)
    
    D = np.sqrt(np.diag (np.diag(S)))
    D_L = np.linalg.inv(D)
    R=D_L.dot(S.dot (D_L))
    
    R=np.round(R,2)
    return R

#****************************************************************************************************
# *****Problem 1 initial & test*****
N_p1 = random.randint(2,10)
#use random to decide N for Problem 1
D_p1 = random.randint(2,10)
#use random to decide D for Problem 1
Problem1(N_p1, D_p1)
print('')  
print('****************************************************************************************************')
print('')  

# *****Problem 2 initial & test*****
N_p2=0
D_p2=1
while N_p2<D_p2:
    #if N>D, the index is not enough
    N_p2 = random.randint(2,10)
    #use random to decide D for Problem 2
    D_p2 = random.randint(2,10)
    #use random to decide D for Problem 2

Problem2(N_p2,D_p2)
print('')  
print('****************************************************************************************************')
print('')  

# *****Problem 3 initial & test*****

print('Problem 3')

#Distance Matrix
#Iris
X = np.random.randint(0,100, (150,4))
#Iris time with using loop
start_Iris_loop=datetime.now()
#start time
compute_distance_native(X)
#run function
time_Iris_loop=datetime.now()-start_Iris_loop
#total time = end time - start time
#Iris time without using loop
start_Iris_no_loop=datetime.now()
compute_distance_smart(X)
time_Iris_no_loop=datetime.now()-start_Iris_no_loop

#Breast cancer
X = np.random.randint(0,100, (569,30))
#Breast cancer time with using loop
start_Breast_loop=datetime.now()
compute_distance_native(X)
time_Breast_loop=datetime.now()-start_Breast_loop
#Breast cancer time without using loop
start_Breast_no_loop=datetime.now()
compute_distance_smart(X)
time_Breast_no_loop=datetime.now()-start_Breast_no_loop

#Digits
X = np.random.randint(0,100, (569,30))
#Digits time with using loop
start_Digits_loop=datetime.now()
compute_distance_native(X)
time_Digits_loop=datetime.now()-start_Digits_loop
#Digits time without using loop
start_Digits_no_loop=datetime.now()
compute_distance_smart(X)
time_Digits_no_loop=datetime.now()-start_Digits_no_loop

#print Distance Matrix
print('Distance Matrix')
table = [["use",time_Iris_loop,time_Breast_loop,time_Digits_loop],["without using",time_Iris_no_loop,time_Breast_no_loop,time_Digits_no_loop]]
headers = ["use loop or not", "Iris","Breast cancer","Digits"]
print tabulate(table, headers, tablefmt="plain")
print('')  

#******************************

#Correlation Matrix
#Iris
X = np.random.randint(0,100, (150,4))
#***Iris time with using loop
start_Iris_loop=datetime.now()
compute_correlation_native(X)
time_Iris_loop=datetime.now()-start_Iris_loop
#Iris time without using loop
start_Iris_no_loop=datetime.now()
compute_correlation_smart(X)
time_Iris_no_loop=datetime.now()-start_Iris_no_loop

#Breast cancer
X = np.random.randint(0,100, (569,30))
#Breast cancer time with using loop
start_Breast_loop=datetime.now()
compute_correlation_native(X)
time_Breast_loop=datetime.now()-start_Breast_loop
#Breast cancer time without using loop
start_Breast_no_loop=datetime.now()
compute_correlation_smart(X)
time_Breast_no_loop=datetime.now()-start_Breast_no_loop

#Digits
X = np.random.randint(0,100, (569,30))
#Digits time with using loop
start_Digits_loop=datetime.now()
compute_correlation_native(X)
time_Digits_loop=datetime.now()-start_Digits_loop
#Digits time without using loop
start_Digits_no_loop=datetime.now()
compute_correlation_smart(X)
time_Digits_no_loop=datetime.now()-start_Digits_no_loop

print('Correlation Matrix')
table = [["use",time_Iris_loop,time_Breast_loop,time_Digits_loop],["without using",time_Iris_no_loop,time_Breast_no_loop,time_Digits_no_loop]]
headers = ["use loop or not", "Iris","Breast cancer","Digits"]
print tabulate(table, headers, tablefmt="plain")

