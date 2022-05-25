import RO
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
import random as r
import gurobipy as gp
from gurobipy import GRB
import csv
import torch
import sys
import pandas
from sklearn import preprocessing
import random
import os

alpha=float(sys.argv[1])
val=int(sys.argv[2])
n_class=int(sys.argv[3])
t=int(sys.argv[4])
l=int(sys.argv[5])
s=int(sys.argv[6])
alp_bisection=int(sys.argv[7])

'''
alpha=0.5
val=0
n_class=2
t=10
l=10
s=10
alp_bisection=0
'''
datadir="scripts/selectiondata16/"
resultdir="path/mine_soft_assign_AE1_0.01_2_alpha"+str(alpha)+'/'
analysisdir = "scripts/results/"

def scaler(df):
    x=df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df


def testPolicyVaR(returns,delta):
    tmp = sorted(returns)
    VaR=tmp[int(np.floor(delta*len(returns)))]
    return VaR

def testPolicyVaR_new(returns,delta):
    tmp = sorted(returns)
    VaR=tmp[int(np.floor(delta*len(returns))) ]
    
    return VaR


def backtest_policy(val,VaR):
    # data=pd.DataFrame(data)
    x=[1 for i in val if i>VaR]
    return float(np.mean(x)),x


def get_assignnments(test_assignments):
    test_assignments_t = test_assignments.transpose()    
    numberList = range(len(test_assignments))
    test_ass_1 = []
    for i in range(len(test_assignments_t)):
        test_ass_1.append(random.choices(numberList, weights= test_assignments_t[i], k=1)[0])
    return np.array(test_ass_1)



def bisection(f,a,b,N):
    if f(a)*f(b) >= 0:
        print("Bisection method fails.")
        return None
    a_n = a
    b_n = b
    for n in range(1,N+1):
        m_n = (a_n + b_n)/2
        f_m_n = f(m_n)
        if f(a_n)*f_m_n < 0:
            a_n = a_n
            b_n = m_n
        elif f(b_n)*f_m_n < 0:
            a_n = m_n
            b_n = b_n
        elif f_m_n == 0:
            # print("Found exact solution.")
            return m_n
        else:
            # print("Bisection method fails.")
            return None
    return (a_n + b_n)/2

def get_alpha(df, mu, c, radius, delta,sig = None):
    alpha_list = []
        
    for i in range(len(df)):
        if sig is not None:
            a=df[i].reshape(1,-1) - mu.reshape(1,-1)*(1-0.1) - c.reshape(1,-1)*(0.1)
            b=df[i].reshape(-1,1) - (1-0.1)*mu.reshape(-1,1) - (0.1)*c.reshape(-1,1)
            f = lambda x:  np.matmul(np.matmul((1/x)*(df[i].reshape(1,-1) - mu.reshape(1,-1)) +mu.reshape(1,-1) - c.reshape(1,-1),sig), ( (1/x)*(df[i].reshape(-1,1) - mu.reshape(-1,1)) - c.reshape(-1,1)))[0][0] - radius**2
            
        else:
            f = lambda x:  np.linalg.norm((1/x)*(df[i] - mu) + mu - c, 2) - radius
        
        approx_alpha = bisection(f,-0.01, 1,1000)  #0.0000001
        alpha_list.append(approx_alpha)
    
        # print(approx_alpha)
    
    alpha_list = list(filter(None, alpha_list))
    alpha_sorted = sorted(alpha_list)
    for alp in alpha_sorted:
        distlist = []
        if sig is not None:
            f = lambda x:  np.matmul(np.matmul((1/x)*(df[i].reshape(1,-1) - mu.reshape(1,-1)) +mu.reshape(1,-1) - c.reshape(1,-1),sig), ( (1/x)*(df[i].reshape(-1,1) - mu.reshape(-1,1)) - c.reshape(-1,1)))[0][0] - radius**2
    
        else:
            f = lambda x:  np.linalg.norm((1/x)*(df[i] - mu) + mu - c, 2) - radius
        
        for i in range(len(df)):
            distlist.append(np.where(f(alp) <= 0, 1, 0).item())
                        
        if (sum(distlist) < int(delta*len(distlist))):
            
            continue
        else:
            break

    return alp
  
    
def get_alpha_for_convex_hull(X_train_main,X_proj, mu, c,sig, radius, delta):
        
    center = X_train_main.mean(axis=0)
        
    N=X_train_main.shape[1]
    Rs=np.transpose(X_train_main)
    (nStocks,nMonths)=Rs.shape

    mu_temp=np.mean(Rs, axis=1)

 
    tmp = Rs - mu_temp.reshape(-1,1)@np.ones((1,nMonths))
    tmp_max = abs(tmp).max(1)

    # Zs = np.diag(1/tmp_max)@tmp
    P = np.diag(tmp_max)
    R_all = np.linalg.norm(tmp,axis=0)
    tmp=np.sort(np.linalg.norm(tmp,axis=0))
    gamma = testPolicyVaR_new(tmp,delta)
    
    outside  = [i for i in range(len(R_all)) if R_all[i] > gamma]
    outside_points = np.take(X_proj, outside, 0)
    
    inside  = [i for i in range(len(R_all)) if R_all[i] <= gamma]
    inside_points = np.take(X_proj, inside, 0)
    
    
    
    alp = get_alpha(inside_points, mu, c, radius, delta,sig)
    
    return alp

fileName = datadir+'test-'+str(t)+'-'+str(l)+'-'+str(s)+'.txt'
X_test = np.genfromtxt(fileName, delimiter=',')
X_test_df=pd.DataFrame(X_test)  #scaler(pd.DataFrame(X_test))
X_test_side=X_test_df.iloc[:,0:2].to_numpy()
X_test_main=X_test_df.iloc[:,2:4].to_numpy()
fileName = datadir+'train-'+str(t)+'-'+str(l)+'-'+str(s)+'.txt'
X_train = np.genfromtxt(fileName, delimiter=',')
X_train_df=pd.DataFrame(X_train)    #scaler(pd.DataFrame(X_train))
X_train_side=X_train_df.iloc[:,0:2].to_numpy()
X_train_main=X_train_df.iloc[:,2:4].to_numpy()
N=X_train_main.shape[1]
df=X_test_df.copy()


with open(resultdir+'test_assignments.npy', 'rb') as f:
    test_assignments = np.load(f)
with open(resultdir+'train_assignments.npy', 'rb') as f:
    train_assignments = np.load(f)
    
mu = np.array(X_train_main.mean(axis=0)).reshape((1, N))

data_assignment = train_assignments.argmax(0)
test_assignment = test_assignments.argmax(0)

data_assignment_rand = get_assignnments(train_assignments)
test_assignment_rand = get_assignnments(test_assignments)
# test_assignment_rand = np.array(random.choices(population=[0,1],weights=[0.5, 0.5],k=251))
print(sum(test_assignment_rand), len(test_assignment_rand))
print(sum(data_assignment_rand), len(data_assignment_rand))


assignments_df = pd.DataFrame(test_assignments).T
assignments_df['max_assign'] = test_assignment
assignments_df['random_assign'] = test_assignment_rand
assignments_df.to_csv(resultdir+'assignments_df'+str(alp_bisection)+'.csv')

prefixed = [filename for filename in os.listdir(resultdir) if filename.startswith("c_")]
n_class=len(prefixed)

#if pobability distribution is equal, randomly assign clusters
if len(set(data_assignment))!=n_class:
    data_assignment = np.array(random.choices(range(n_class), k=train_assignments.shape[1]))
    test_assignment = np.array(random.choices(range(n_class), k=test_assignments.shape[1]))


for alp_bisection in [-1000,0]:

    # df_new=pd.concat([X_test_df.copy(),pd.DataFrame(test_assignments).T,pd.DataFrame(test_assignment)],axis=1)
    df_rand_new=pd.concat([X_test_df.copy(),pd.DataFrame(test_assignments).T,pd.DataFrame(test_assignment_rand)],axis=1)

    df=pd.concat([X_test_df,pd.DataFrame(test_assignments).T,pd.DataFrame(test_assignment)],axis=1)
    df_rand=pd.concat([X_train_df.copy(),pd.DataFrame(train_assignments).T,pd.DataFrame(data_assignment_rand)],axis=1)
    df_adj_IDCC=pd.concat([X_test_df,pd.DataFrame(test_assignments).T,pd.DataFrame(test_assignment)],axis=1)
    for delta in [0.60,0.70,0.80,0.90,0.95,0.99]:
        back_test_violations=[]
        R_mark=[]
        R_bootstrap=[]
        obj_mark=[]
        obj_bootstrap=[]
        test_vals_mark=[]
        test_vals_mark_rand=[]
        test_vals_bootstrap=[]
    
        for k in range(n_class):
            df[str(delta)+'_'+str(k)]=False
            df_rand[str(delta)+'_'+str(k)]=False
            E = 2
            L = 2
            fileName = resultdir+"c_"+str(k)+".txt"
            c0 = np.genfromtxt(fileName, delimiter=',')
            
            fileName = resultdir+"cov_"+str(k)+".txt"
            sig=np.genfromtxt(fileName, delimiter=',')
            sig_inv = np.linalg.inv(sig)
            sig = sig_inv
            print(sig)
            listW = []
            dimLayers = []
            
            for F in range(0,L,1):
                fileName = resultdir+"W_"+str(k)+"_"+str(F)+".txt"
                listW.append(np.genfromtxt(fileName, delimiter=','))
                dimLayers.append(listW[F].shape[0])
                
            N=listW[0].shape[1]
            
            
            maxScenEntry = max(np.amax(X_train),np.amax(X_test))
            maxEntry = max(np.amax(X_train),np.amax(X_test))
            M=[]
            for i in range(0,L,1):
                rowSums = np.sum(np.absolute(listW[i]),axis=1)
                M.append(maxEntry*np.amax(rowSums))
                maxEntry = maxEntry*np.amax(rowSums)
               
            X_train_hat=[]
            for j in range(0,X_train_main[np.where(data_assignment==k)[0],:].shape[0],1):
                outLayer = X_train_main[np.where(data_assignment==k)[0],:][j,:]
                sigma = []
                for i in range(0,L-1,1):
                    sigmal = []
                    for l in range(listW[i].shape[0]):
                        if np.dot(listW[i],outLayer)[l] > 0:
                            sigmal.append(1)
                        else:
                            sigmal.append(0)
                    outLayer = np.maximum(np.zeros(listW[i].shape[0]),np.dot(listW[i],outLayer))
                    # print(i,outLayer)
                    sigma.append(sigmal)
                    
                # sigmas.append(sigma)
                outLayer = np.dot(listW[L-1],outLayer)
                X_train_hat.append(outLayer)
                
            #mu_hat
            outLayer = mu[0]
            for i in range(0,L-1,1):
                sigmal = []
                for l in range(listW[i].shape[0]):
                    if np.dot(listW[i],outLayer)[l] > 0:
                        sigmal.append(1)
                    else:
                        sigmal.append(0)
                outLayer = np.maximum(np.zeros(listW[i].shape[0]),np.dot(listW[i],outLayer))
            mu_hat = np.dot(listW[L-1],outLayer)

            #marks radius
            try:
                R_all_, R,sigmas,lb,ub = RO.getRadiiDataPoints(L,c0,sig, X_train_main[np.where(data_assignment==k)[0],:],listW, 0, delta)

                temp=pd.DataFrame(RO.plotRadiiDataPoints(R,L, c0, sig,X_test_main[np.where(test_assignment==k)[0],:], listW, 0, delta),columns=['x','y',str(delta)])
                
                R_mark=R
                R_all_test, _,_,_,_ = RO.getRadiiDataPoints(L,c0,sig, X_test_main[np.where(test_assignment==k)[0],:],listW, 0, delta)
                
                df.loc[test_assignment==k, str(delta)+'_'+str(k)]=list(R_all_test<=R)
                
                Rand_R_all_, Rand_R,Rand_sigmas,Rand_lb,Rand_ub = RO.getRadiiDataPoints(L,c0,sig, X_train_main[np.where(data_assignment_rand==k)[0],:],listW, 0, delta)

                temp=pd.DataFrame(RO.plotRadiiDataPoints(R,L, c0, sig,X_test_main[np.where(test_assignment_rand==k)[0],:], listW, 0, delta),columns=['x','y',str(delta)])
                df_rand.loc[data_assignment_rand==k, str(delta)+'_'+str(k)]=list(Rand_R_all_<= Rand_R)
                X_train_temp = np.array(X_train_hat)
                # X_train_assigned = X_train_main[np.where(data_assignment_rand==k)[0],:]
                X_train_assigned = X_train_main[np.where(data_assignment==k)[0],:]
        
                outside  = [i for i in range(len(Rand_R_all_)) if Rand_R_all_[i] > Rand_R]
                outside_points = np.take(X_train_assigned, outside, 0)
                if alp_bisection==0:
                    _, Rand_R,_,_,_ = RO.getRadiiDataPoints(L,c0,sig,X_train_main[np.where(data_assignment==k)[0],:],listW, 0, 0.99)
                    
                    alp = get_alpha_for_convex_hull(X_train_assigned,X_train_temp,mu_hat, c0,sig, Rand_R, delta)
                    # alp = get_alpha_for_convex_hull(X_train_assigned,X_train_temp,mu_hat, c0,sig, Rand_R, delta)
                    dist_adj_IDCC=[]
                    for pt in X_test_main[np.where(test_assignment==k)[0],:]:
                        dist_adj_IDCC.append(np.linalg.norm((1/alp)*(pt - mu) + mu - c0, 2) - R)
                    df_adj_IDCC.loc[test_assignment==k, str(delta)+'_'+str(k)]=[True if i <0 else False for i in dist_adj_IDCC]
                else:
                
                    alp = -1000
        
                epsilon=delta/n_class
              
                print('Class: {}\t Delta: {:.8f}\t Alpha: {:.8f}\t Radius: {:.8f}\t Rand Radius: {:.8f}'.format(k,delta,alp, R, Rand_R))
                
                start = time.time()
                p=1
                
                obj, x_mark = RO.solveRobustSelection(p,N,L,X_train_main[np.where(data_assignment==k)[0],:],dimLayers,c0, mu_hat, sig, alp, R, listW, M, lb, ub,0, False, 0,sigmas)
                obj_mark=obj
                x_mark = 100*np.round(x_mark, 4)
                
                print("x=",x_mark)
                
                Rand_obj, Rand_x_mark = RO.solveRobustSelection(p,N,L,X_train_main[np.where(data_assignment_rand==k)[0],:], dimLayers,c0, mu_hat, sig, alp, Rand_R, listW, M, lb, ub,0, False, 0,sigmas)
                Rand_obj_mark=Rand_obj
                Rand_x_mark = 100*np.round(Rand_x_mark, 4)
                print("Rand x=",Rand_obj_mark)
                end = time.time()
                t = end-start
                
                avg, maximum, vals = RO.evaluateSolution(x_mark,X_test_main[np.where(test_assignment==k)[0],:])
                test_vals_mark.extend(vals) 
                
                Rand_avg, Rand_maximum, Rand_vals = RO.evaluateSolution(Rand_x_mark,X_test_main[np.where(test_assignment_rand==k)[0],:])
                test_vals_mark_rand.extend(Rand_vals)    
            except:
                continue
    
            
            b_test_cluster,back_test=backtest_policy(vals,obj)
            back_test_violations.extend(back_test)
            csvFile = open(analysisdir+"Simulation-solutions_val_cluster.csv", 'a')
            out = csv.writer(csvFile,delimiter=",", lineterminator = "\n")
            row = [alpha,delta,k]
            row.append(R_mark)

            row.append(obj_mark)
            row.append(x_mark)
            row.append(avg)
            row.append(maximum)
            row.append(b_test_cluster)
           
            out.writerow(row)
            csvFile.close()
            
        csvFile = open(analysisdir+"Simulation-solutions_val_overall.csv", 'a')
        out = csv.writer(csvFile,delimiter=",", lineterminator = "\n")
        row = [alpha,delta]
        row.append(testPolicyVaR(test_vals_mark,delta))
        for i in range(len(test_vals_mark)):
            row.append(test_vals_mark[i])

        row.append(np.sum(back_test_violations)/len(test_vals_mark))
        out.writerow(row)
        csvFile.close()
                       
    df['SSIZE']=s
    df.to_csv(analysisdir+"IDCC_plot_data_"+str(alp_bisection)+".csv",index=False)
    
    df_rand['SSIZE']=s
    df_rand.to_csv(analysisdir+"IDCC_randplot_data_"+str(alp_bisection)+".csv",index=False)                   
   