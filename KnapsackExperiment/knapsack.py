import torch 
from torch import nn, optim
from torch.utils.data.dataset import Dataset
import math
import numpy as np 
import gurobipy as gp
import os

from torch import nn, optim




def gurobi_knapsack(y, weights,capacity,n_items,relaxation=True):
    vtype = gp.GRB.CONTINUOUS if relaxation else gp.GRB.BINARY
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    x = model.addMVar(shape= n_items, lb=0.0, ub=1.0,vtype=vtype, name="x")
    model.addConstr(weights @ x <= capacity, name="eq")
    model.setObjective(y@x, gp.GRB.MAXIMIZE)
    model.optimize()
    return x.X

def get_qpt_mat(weights, capacity, n_items):
    G = weights
    h = np.array([capacity])
    A = None
    b = None
    return A,b, G,h



class MyCustomDataset(Dataset):
    def __init__(self, X,y,weights,capacity,n_items,relaxed=True):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

        
        sol = []
        for i in range(len(y)):
            x_sol = gurobi_knapsack(y[i],
               weights,capacity,n_items,relaxed=relaxed)            
            sol.append( x_sol)
        self.sol = np.array(sol)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx],self.y[idx],self.sol[idx]
