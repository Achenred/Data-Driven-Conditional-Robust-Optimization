import numpy as np
import matplotlib.pyplot as plt
import math
import random as r
import gurobipy as gp
from gurobipy import GRB
import csv
import sys

from joblib import Parallel, delayed
import multiprocessing
import pandas as pd


def evaluateSolution(x,scenarios):
    maximumValue = -10000000
    averageValue = 0
    vals = []
    
    for i in range(0,scenarios.shape[0],1):
        val = np.dot(x,scenarios[i])
        vals.append(val)
        averageValue = averageValue + val
        if val>maximumValue:
            maximumValue = val
            
    averageValue = averageValue / scenarios.shape[0]
        
    return averageValue, maximumValue, vals

def solveRobustSelectionDiscrete(p,N,scenarios):
    timeLimit = 86400
    gap=0.0001 
     # Create a new model
    ip = gp.Model("AdversarialProblem")
    ip.setParam("OutputFlag", 0)
    
    ip.setParam("TimeLimit", timeLimit)
    ip.setParam('MIPGap', gap)
    x = ip.addVars(N, vtype=GRB.CONTINUOUS, lb = 0.0, ub = 1.0, name="x")
    z = ip.addVar(vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY,obj=1.0, name="z")
    
    #Add Sum x_i = p constraint
    lhs=""
    for i in range(0,N,1):
        lhs = lhs + 1 * x[i]
        
    ip.addConstr(lhs, sense=GRB.EQUAL, rhs=p, name="x_equ_p")
    
    
    for j in range(0,scenarios.shape[0],1):
        
        #Add new constraint c^T x
        lhs = ""
        for i in range(0,N,1):
            lhs = lhs + scenarios[j,i] * x[i]
         
        lhs = lhs - z
        ip.addConstr(lhs, sense=GRB.LESS_EQUAL, rhs=0, name="c"+str(j))

        
    ip.optimize()

    x_opt = np.zeros(N)
    for i in range(0,N,1):
        x_opt[i] = ip.getVarByName(x[i].VarName).x
    
    obj = ip.objVal
            
    return obj, x_opt

#Gurobi will default minimize the objective
def solveRobustSelection(p,N,L,dimLayers,c0, R, W, M, lb, ub, ReLUType=0, twoStage=False, c1St=0, sigmas=[0]):
    timeLimit = 86400
    gap=0.0001 
    eps = 0.01
     # Create a new model
    ip = gp.Model("AdversarialProblem")
    ip.setParam("OutputFlag", 0)
    
    ip.setParam("TimeLimit", timeLimit)
    ip.setParam('MIPGap', gap)
    if twoStage==True:
        x = ip.addVars(N, obj=c1St, vtype=GRB.BINARY, name="x")
    else:
        x = ip.addVars(N, vtype=GRB.CONTINUOUS,lb=0.0,ub = 1.0, name="x")
    mu = ip.addVar(vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY,obj=1.0, name="mu")
    
    #Add Sum x_i = p constraint
    lhs=""
    for i in range(0,N,1):
        lhs = lhs + 1 * x[i]
    
    if twoStage == True:
        ip.addConstr(lhs, sense=GRB.LESS_EQUAL, rhs=p, name="x_equ_p")
    else:
        ip.addConstr(lhs, sense=GRB.EQUAL, rhs=p, name="x_equ_p")
    
    
    #Select feasible solution
    x_opt = [0.2, 0.0, 0.0, 0.0, 0., 0., 0., 0., 0., 0., 0.1, 0.2, 0.5]
#np.array([1/N]*N)
    
    notOptimal = True
    obj = -10
    objWC = 0
    counter = 0
    while abs(objWC-obj) >= eps*abs(obj):
        if counter%10 == 0:
            print("Counter = ", counter)
        
        if counter > 0:
            ip.optimize()
            # print(ip.getConstrs()[0])
            # ip.write(r'/Users/nbandi/Dropbox/Mac/Desktop/PhD/Research/Robust/RO-DNN-master/code/solver/gurobi_model.mps')
            x_opt = np.zeros(N)
            for i in range(0,N,1):
                x_opt[i] = ip.getVarByName(x[i].VarName).x
            
            obj = ip.objVal
        
        if counter%10 == 0:
            print("x=",x_opt)
        
        counter = counter + 1
        if twoStage == True:
            probType = "2StageSelection"
            objWC, z, a = getWorstCaseScenario(N,L,dimLayers,x_opt,c0, R, W, M, lb, ub, ReLUType, probType, p)
            print("z=",z)
            print("a=",a)
        else:
            probType = ""
            objWC, c = getWorstCaseScenarioSigmas(N,L,dimLayers,x_opt,c0, R, W, M, lb, ub, ReLUType, probType, p, sigmas)
            
        if twoStage == True:
            objWC = objWC + np.dot(x_opt,c1St)
        

        #Add new constraint c^T x
        lhs = ""
        for i in range(0,N,1):
            if twoStage == True:
                lhs = lhs + (z[i]-a) * x[i]
            else:
                lhs = lhs + c[i] * x[i]
         
        lhs = lhs - mu
        if twoStage:
            ip.addConstr(lhs, sense=GRB.LESS_EQUAL, rhs=sum(z)-(a*p), name="z"+str(counter))
        else:
            ip.addConstr(lhs, sense=GRB.LESS_EQUAL, rhs=0, name="c"+str(counter))
  
        if counter%10 == 0:
            print("#####VALUES", obj, objWC)
        
            
    return obj, x_opt

def getWorstCaseScenarioSigma(N,L,dimLayers,x,c0, R, W, M, lb, ub, ReLUType=0, probType=0, p=0, sigma=[0]):
        
        timeLimit = 86400
        gap=0.01 
        reluAlpha = 0.1
  
  
        ip = gp.Model("AdversarialProblem")
        ip.setParam("OutputFlag", 0)
    
        ip.setParam("TimeLimit", timeLimit)
        ip.setParam('MIPGap', gap)
        ip.setParam("FeasibilityTol", 0.000000001)

        # Create variables
        c = []

        c.append(ip.addVars(N, lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name="c[1]"))
    
        for i in range(1,L,1):
            c.append(ip.addVars(dimLayers[i-1], lb=0, vtype=GRB.CONTINUOUS, name="c["+ str(i+1) + "]"))
        c.append(ip.addVars(dimLayers[L-1], lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="c["+ str(L) + "]"))
        
        xi = ip.addVars(dimLayers[L-1], lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="xi")
    
        #Set objective
        lhs=""
        for i in range(0,N,1):
            lhs = lhs + x[i] * c[0][i]
                
        ip.setObjective(lhs, GRB.MAXIMIZE)
        

        for i in [L-1]:
            for l in range(0,dimLayers[i],1):
                dimL = dimLayers[i-1]
                lhs=""
                lhs = lhs + 1 * c[i+1][l]
                for j in range(0,dimL,1):
                    lhs = lhs - W[i][l,j] * c[i][j]
                                
                ip.addConstr(lhs, sense=GRB.EQUAL, rhs=0.0, name="c"+str(i+1)+"_" + str(l))


        for i in range(0,L-1,1):
            for l in range(0,dimLayers[i],1):
                if i==0:
                    dimL = N
                else:
                    dimL = dimLayers[i-1]

                lhs=""
                for j in range(0,dimL,1):
                    if sigma[i][l] > 0.5:
                        lhs = lhs + W[i][l,j] * c[i][j]
                    else:
                        lhs = lhs - W[i][l,j] * c[i][j]
                
                ip.addConstr(lhs, sense=GRB.GREATER_EQUAL, rhs=0.0, name="W"+str(i+1)+"_L_" + str(l))
                
                if sigma[i][l] > 0.5:
                    lhs=""
                    lhs = lhs + 1*c[i+1][l]
                    for j in range(0,dimL,1):
                        lhs = lhs - W[i][l,j] * c[i][j]
                                    
                    ip.addConstr(lhs, sense=GRB.EQUAL, rhs=0.0, name="c"+str(i+1)+"_" + str(l))
                    
                else:
                    ip.addConstr(c[i+1][l], sense=GRB.EQUAL, rhs=0.0, name="c"+str(i+1)+"_" + str(l))
    
        #Add Ball-Constraint    
        lhs = ""
        for i in range(0,dimLayers[L-1],1):
            lhs = lhs + xi[i] * xi[i]
        
        ip.addQConstr(lhs, sense=GRB.LESS_EQUAL, rhs=R*R, name="Ball")
        
        #Add xi = c-c0 constraints
        for j in range(0,dimLayers[L-1],1):
            lhs = ""
            lhs = 1 * c[L][j] - 1 * xi[j] 
            ip.addConstr(lhs, sense=GRB.EQUAL, rhs=c0[j], name="xi" + str(j))
            
        
        ip.optimize()
        
        
        c_ret = np.zeros(N)
        for i in range(0,N,1):
            c_ret[i] = ip.getVarByName(c[0][i].VarName).x
            
        return ip.objVal, c_ret 


def getWorstCaseScenarioSigmas(N,L,dimLayers,x,c0, R, W, M, lb, ub, ReLUType=0, probType=0, p=0, sigmas=[0]):
    
    
    objvals = []
    c_opts = []
    
    objvals, c_opts = zip(*Parallel(n_jobs=1)(delayed(getWorstCaseScenarioSigma)(N,L,dimLayers,x,c0, R, W, M, lb, ub, ReLUType, probType, p, sigma) for sigma in sigmas))

    
    return np.max(objvals),c_opts[np.argmax(objvals)]
        

def getWorstCaseScenario(N,L,dimLayers,x,c0, R, W, M, lb, ub, ReLUType=0, probType=0, p=0):
    timeLimit = 86400
    gap=0.0001 
    reluAlpha = 0.1
    
    # Create a new model
    ip = gp.Model("AdversarialProblem")
    ip.setParam("OutputFlag", 0)
    
    ip.setParam("TimeLimit", timeLimit)
    ip.setParam('MIPGap', gap)
    ip.setParam("FeasibilityTol", 0.000000001)
    
    # Create variables
    u = []
    c = []

    c.append(ip.addVars(N, lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name="c[1]"))
    
    for i in range(1,L,1):
        u.append(ip.addVars(dimLayers[i-1], vtype=GRB.BINARY, name="u["+ str(i) + "]"))
    
    for i in range(1,L,1):
        c.append(ip.addVars(dimLayers[i-1], lb=0, vtype=GRB.CONTINUOUS, name="c["+ str(i+1) + "]"))
    c.append(ip.addVars(dimLayers[L-1], lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="c["+ str(i+1) + "]"))
    
    xi = ip.addVars(dimLayers[L-1], lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="xi")
    
    if probType == "2StageSelection":
        z = ip.addVars(N, vtype=GRB.CONTINUOUS, name="z")
        a = ip.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="a")
    
    #Set objective
    lhs=""

    if probType == "2StageSelection":
        lhs = ""
        sumX = 0
        for i in range(0,N,1):
            lhs = lhs + (x[i]-1) * z[i]
            sumX = sumX + x[i]
        lhs = lhs + (p-sumX) * a
    else:
        for i in range(0,N,1):
            lhs = lhs + x[i] * c[0][i]
            
    ip.setObjective(lhs, GRB.MAXIMIZE)
    

    for i in [L-1]:
        for l in range(0,dimLayers[i],1):
            dimL = dimLayers[i-1]
            lhs=""
            lhs = lhs + 1 * c[i+1][l]
            for j in range(0,dimL,1):
                lhs = lhs - W[i][l,j] * c[i][j]
            
            ip.addConstr(lhs, sense=GRB.EQUAL, rhs=0.0, name="c"+str(i+1)+"_" + str(l))


    for i in range(0,L-1,1):
        for l in range(0,dimLayers[i],1):
            if i==0:
                dimL = N
            else:
                dimL = dimLayers[i-1]

            
            lhs=""
            for j in range(0,dimL,1):
                lhs = lhs + W[i][l,j] * c[i][j] * (u[i][l]-1)
                
            ip.addQConstr(lhs, sense=GRB.GREATER_EQUAL, rhs=0.0, name="W"+str(i+1)+"_L_" + str(l))
            
            
            
            
            if ReLUType==0:
                #Classical ReLU
                lhs=""
                lhs = lhs + 1 * c[i+1][l]
                for j in range(0,dimL,1):
                    lhs = lhs - W[i][l,j] * c[i][j] * u[i][l] 
                
                ip.addQConstr(lhs, sense=GRB.EQUAL, rhs=0.0, name="c"+str(i+1)+"_" + str(l))
                
            elif ReLUType == 1:
                #Leaky ReLU
                lhs=""
                lhs = lhs + 1 * c[i+1][l]
                for j in range(0,dimL,1):
                    lhs = lhs - (1-reluAlpha)*W[i][l,j] * c[i][j] * u[i][l] - reluAlpha*W[i][l,j] * c[i][j]
                
                ip.addQConstr(lhs, sense=GRB.EQUAL, rhs=0.0, name="c"+str(i+1)+"_" + str(l))
      
    #Add Ball-Constraint    
    lhs = ""
    for i in range(0,dimLayers[L-1],1):
        lhs = lhs + xi[i] * xi[i]
    
    ip.addQConstr(lhs, sense=GRB.LESS_EQUAL, rhs=R*R, name="Ball")
    
    #Add xi = c-c0 constraints
    for j in range(0,dimLayers[L-1],1):
        lhs = ""
        lhs = 1 * c[L][j] - 1 * xi[j] 
        ip.addConstr(lhs, sense=GRB.EQUAL, rhs=c0[j], name="xi" + str(j))
        
    if probType=="2StageSelection":
        for i in range(0,N,1):
            lhs = a - z[i] - c[0][i]
            ip.addConstr(lhs, sense=GRB.LESS_EQUAL, rhs=0, name="z" + str(j))
    
    # Optimize model
    ip.optimize()
    
    if ip.status  == GRB.OPTIMAL:
        print('Model is optimal')
    elif ip.status  == GRB.INFEASIBLE:
        print('Model  is  infeasible')
    elif ip.status  == GRB.UNBOUNDED:
        print('Model  is  unbounded')
    else:
        print('Optimization  ended  with  status ' + str(ip.status))
    
    objVal = ip.objVal
    
    if probType == "2StageSelection":
        z_ret = np.zeros(N)
        a_ret=0
        for i in range(0,N,1):
            z_ret[i] = ip.getVarByName(z[i].VarName).x
        a_ret = ip.getVarByName(a.VarName).x
        return objVal,z_ret,a_ret
    else:
        c_ret = np.zeros(N)
        for i in range(0,N,1):
            c_ret[i] = ip.getVarByName(c[0][i].VarName).x
                        
        return objVal, c_ret
        


def getCorePoints(L, c0, X, listW, reluType, quantil):
    R = []
    for j in range(0,X.shape[0],1):
        outLayer = X[j,:]
        for i in range(0,L-1,1):
            outLayer = np.maximum(np.zeros(listW[i].shape[0]),np.dot(listW[i],outLayer))
        outLayer = np.dot(listW[L-1],outLayer)
            
        R.append(np.linalg.norm(outLayer-c0))
    
    Rquant = np.nanquantile(R,quantil)
    
    X_ret = [X[i,:] for i in range(0,X.shape[0],1) if R[i] <= Rquant]
    
    return np.array(X_ret)

def getCorePoints_kern(p,N,X,Q,theta,alphas,SV):
    
    R = []
    for j in range(0,X.shape[0],1):
        R.append(np.sum([alphas[k]*np.linalg.norm(Q.dot(X[j,:]-X[k,:]), ord=1) for k in SV]))
    
    X_ret = [X[i,:] for i in range(0,X.shape[0],1) if R[i] <= theta]
    
    return np.array(X_ret)
    
    
def plotRadiiDataPoints(L, c0, X, listW, reluType, quantil, A, C):

    R = []
    sigmas = []
    for j in range(0,X.shape[0],1):
        outLayer = X[j,:]
        sigma = []
        for i in range(0,L-1,1):
            sigmal = []
            for l in range(listW[i].shape[0]):
                if np.dot(listW[i],outLayer)[l] > 0:
                    sigmal.append(1)
                else:
                    sigmal.append(0)
            outLayer = np.maximum(np.zeros(listW[i].shape[0]),np.dot(listW[i],outLayer))
            sigma.append(sigmal)
            
        sigmas.append(sigma)
        outLayer = np.dot(listW[L-1],outLayer)
            
        R.append(np.linalg.norm(outLayer-c0))
    
    Rquant = np.quantile(R,quantil)
    
    sigmas = [sigmas[i] for i in range(len(sigmas)) if R[i] <= Rquant]
    
    sigmas = [x for i, x in enumerate(sigmas) if i == sigmas.index(x)]
    
    for a in np.linspace(0.0, 100.0, num=250):
        for b in np.linspace(0.0, 100.0, num=250):
            feas = False
            for sigma in sigmas:
                sfeas = True
                outLayer = [a,b]
                for i in range(0,L-1,1):
                    outLayer = np.dot(listW[i],outLayer)
                    for l in range(listW[i].shape[0]):                        
                        if sigma[i][l] > 0.5 and outLayer[l] < 0:
                            sfeas = False
                        if sigma[i][l] < 0.5 and outLayer[l] > 0:
                            sfeas = False
                        if sigma[i][l] < 0.5 and outLayer[l] < 0:
                            outLayer[l] = 0
                outLayer = np.dot(listW[L-1],outLayer)
                r = np.linalg.norm(outLayer-c0)
                
                if sfeas == True and r <= Rquant:
                    feas = True
            
            if feas == True:
                print ("plot-"+str(A)+"-"+str(C), a,b,1)
            else:
                print ("plot-"+str(A)+"-"+str(C), a,b,0)
 

def getRadiiDataPoints(L, c0, X, listW, reluType, quantil):
    maxRadius = 0
    R = []
    sigmas = []
    for j in range(0,X.shape[0],1):
        outLayer = X[j,:]
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
            
        sigmas.append(sigma)
        outLayer = np.dot(listW[L-1],outLayer)
        # print(X[i],outLayer,np.linalg.norm(outLayer-c0))
        R.append(np.linalg.norm(outLayer-c0))
    
    Rquant = np.nanquantile(R,quantil)
    
    sigmas = [sigmas[i] for i in range(len(sigmas)) if R[i] <= Rquant]
    
    sigmas = [x for i, x in enumerate(sigmas) if i == sigmas.index(x)]
    
    
    print("Sigmas: ", len(sigmas))
    
    
    N=listW[0].shape[1]
    lb = np.zeros(N)
    ub = np.zeros(N)
    for j in range(0,X.shape[1],1):
        lb[j] = np.amin([ X[i,j] for i in range(len(X[:,j])) if R[i] <= Rquant])
        ub[j] = np.amax([ X[i,j] for i in range(len(X[:,j])) if R[i] <= Rquant])
                        
                  
            
    return R, Rquant,sigmas,lb,ub

def randomiser(assignment_perc, size,used_list):
    while True:
        l= np.random.binomial(1, p= assignment_perc, size=[size])
        if tuple(l) not in used_list:
            return l
        

def bootstrap_radius(dist,prob, epsilon,delta):
    all_radii=[]
    for ass_perc in np.arange(epsilon+0.1, delta, 0.05):
        print('percent : '+str(ass_perc))
        temp_df=pd.DataFrame({'dist':dist,'prob':prob}).sort_values(by='prob', ascending=False)
    
        # temp_df = param_dist[[dist_1, prob_1]].sort_values(by=[prob_1], ascending=False)
        temp_df['cum_sum'] = temp_df['prob'].cumsum()
        temp_df['cum_perc'] = round(100*temp_df.cum_sum/temp_df['prob'].sum(),2)
    
        ind = temp_df[temp_df.cum_perc <= epsilon*100].tail(1).index[0]
        threshold = epsilon*temp_df.prob.sum()
    
        used_list = []
        useful_list = []
        radii_list = []
        for i in range(10000):
            used_array = randomiser(assignment_perc = ass_perc, size = temp_df.shape[0],used_list=used_list)
            temp_df['assign'] =  used_array    #temp_df.apply(lambda x: randomiser(x), axis=1)
      
            # temp_df.loc[:ind,'assign'] = 1
            temp_df["prod"] = temp_df['prob'] * temp_df["assign"]
      
            if (temp_df["prod"].sum()) >= threshold:
                df_get_radius = temp_df.loc[temp_df['assign'] == 1][['dist', 'prob']].sort_values(by=['dist','prob'], ascending=[True,False])
                df_get_radius['cum_sum'] = df_get_radius['prob'].cumsum()
                radii_list.append(df_get_radius[ (df_get_radius.cum_sum <= threshold)]['dist'].max())
        
                useful_list.append(tuple(used_array))
                # if df_get_radius[ (df_get_radius.cum_sum <= threshold)].tail(1)['dist'].iloc[0]>0.5:
                #     print(temp_df["prod"].sum(),temp_df)
                #     break
            # if (i > 1):
            #     if (tuple(used_array) in used_list):
            #         continue
              
            used_list.append(tuple(used_array))
            
            temp_df.drop(['assign', 'prod'], axis=1, inplace=True)

                
            if i%5000==0:
                print('iter : '+str(i))
            # if len(radii_list) == 1000:
            #   break
        all_radii.extend(radii_list)
        if len(radii_list)>0:
            print(ass_perc,max(radii_list),np.mean(radii_list))
    return all_radii,np.nanquantile(all_radii,delta)
        



def solveKernel(p,N,scenarios,Q,theta,alphas,SV):    
    
    #N = problem dimension
    #K = scenarios size
    K = len(scenarios)
    cSV = len(SV)
    
    timeLimit = 86400
    gap=0.0001 
     # Create a new model
    ip = gp.Model("Kernel")
    ip.setParam("OutputFlag", 0)
    
    ip.setParam("TimeLimit", timeLimit)
    ip.setParam('MIPGap', gap)

    x = ip.addVars(N, vtype=GRB.CONTINUOUS, lb = -1.0, ub = 1.0, name="x")

    mu = []
    for i in range(cSV):
        mu.append(ip.addVars(N, vtype=GRB.CONTINUOUS, obj=Q.dot(scenarios[SV[i]])))
    
    lamb = []
    for i in range(cSV):
        lamb.append(ip.addVars(N, vtype=GRB.CONTINUOUS, obj=-Q.dot(scenarios[SV[i]])))
        
    eta=ip.addVar(vtype=GRB.CONTINUOUS, obj=theta)
    
    
    #Add Sum x_i = p constraint
    lhs=""
    for i in range(0,N,1):
        lhs = lhs + 1 * x[i]
        
    ip.addConstr(lhs, sense=GRB.EQUAL, rhs=p, name="x_equ_p")
    
    
    for j in range(N):
        lhs = ""
        for i in range(cSV):
            for l in range(N):
                lhs = lhs + Q[j][l]*(lamb[i][l] - mu[i][l])
         
        lhs = lhs + x[j]
        ip.addConstr(lhs, sense=GRB.EQUAL, rhs=0, name="c"+str(j))

    for i in range(cSV):
        for j in range(N):
            lhs = ""
            lhs = lhs + 1*lamb[i][j]
            lhs = lhs + 1*mu[i][j]
            lhs = lhs - alphas[SV[i]]*eta
            ip.addConstr(lhs, sense=GRB.EQUAL, rhs=0, name="p"+str(j))

    
    ip.optimize()
    
    x_opt = np.zeros(N)
    for i in range(0,N,1):
        x_opt[i] = ip.getVarByName(x[i].VarName).x
    
    obj = ip.objVal
            
    return obj, x_opt    
    

