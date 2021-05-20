import sys
sys.path.insert(0, '../')

import math
import numpy as np
import pandas as pd
import time
import argparse
from knapsack import gurobi_knapsack, get_qpt_mat
from methods import SPOSL, SPO, SPO_pool, BB,BB_pool, TwoStage, QPTL, Interior
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse
import sklearn
from utils import launch_jobs
import torch 
from torch import nn, optim



torun = {k:True for k in ['ALL', 'RELU', 'MAP', 'SPO', 'BB', 'QPTL', 'IP', 'BBP','SPOP','2S']}

parser = argparse.ArgumentParser(description='Knapsack test script')
parser.add_argument('--growth', type=float, default=1.0, metavar='P',
                        help='p solve ')
parser.add_argument('--workers', type=int, default=1, metavar='N',
                        help='parallel jobs')

parser.add_argument('--folds', type=int, default=1, metavar='N',
                        help='number of iterations')

parser.add_argument('--out', type=str, default='knapsack.csv', metavar='PATH',
                        help='Path to output csv file')

parser.add_argument('--valid', action='store_true', default=False,
                        help='enable validation')  

parser.add_argument('--seed', type=int, default=None,metavar='S',
                        help='seed')                               

parser.add_argument('--models', nargs='*', metavar="M",
                        help='model(s) to run among {}'.format(list(torun.keys())))  
args = parser.parse_args()

for model in torun.keys():
    torun[model] = model in args.models
print(torun)

g = args.growth

fixed_params = {
    'relax':False,
    'solver':gurobi_knapsack,
    'optimizer': optim.Adam,
    'growth':g,
    'save':True,
    'maximize':True,
    'is_reg':True,
    'unique_only':True,
    'verbose':False,
    'bsizeCOP':24,
    'seed':args.seed,
}

data = np.load('../data/Data.npz')
x_train,  x_test, y_train,y_test = data['X_1gtrain'],data['X_1gtest'],data['y_train'],data['y_test']
x_train = x_train[:,1:]
x_test = x_test[:,1:]
x_valid, x_test = x_test[0:2880,:], x_test[2880:,:]
y_valid, y_test = y_test[0:2880], y_test[2880:]

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1,48,x_train.shape[1])
y_train = y_train.reshape(-1,48)
x_test = x_test.reshape(-1,48,x_test.shape[1])
y_test = y_test.reshape(-1,48)
x = np.concatenate((x_train, x_test), axis=0)
y = np.concatenate((y_train,y_test), axis=0)
x,y = sklearn.utils.shuffle(x,y,random_state=0)
weights = [data['weights'].tolist()]
weights = np.array(weights)
n_items = 48



nce_params = {#(prob, variant):(lr,epochs)
    # w60
    ('w60',1):(0.1, 20),  ('w60',2):(0.01, 20),  ('w60',3):(0.1, 20), 
    ('w60',4):(1e-2, 20),  ('w60',5):(0.7, 20),  ('w60',6):(0.7, 20), 
    ('w60',7):(1e-2, 20),  ('w60',8):(0.7, 20),  ('w60',9):(0.7, 20), 
    # w120
    ('w120',1):(1e-3,20), ('w120',2):(1e-3,20), ('w120',3):(1e-3,20),
    ('w120',4):(1e-2,20), ('w120',5):(0.7,20), ('w120',6):(0.7,20),
    ('w120',7):(1e-3,20), ('w120',8):(0.7,20), ('w120',9):(0.7,20),
    # w180
    ('w180',1):(1e-2,20), ('w180',2):(1e-2,20), ('w180',3):(1e-2,20),
    ('w180',4):(1e-2,20), ('w180',5):(0.7,20), ('w180',6):(0.7,20),
    ('w180',7):(1e-3,20), ('w180',8):(0.7,20), ('w180',9):(0.7,20),
}

two_stage_params = {
    'w60':(('lr',0.1), ('epochs',20)),
    'w120':(('lr',0.1), ('epochs',20)),
    'w180':(('lr',0.1), ('epochs',20))
}

spo_params = {
    'w60':(('lr',0.7), ('epochs',4)),
    'w120':(('lr',0.7), ('epochs',20)),
    'w180':(('lr',0.7), ('epochs',20))
}

spopool_params = {
    'w60':(('lr',0.7), ('epochs',8)),
    'w120':(('lr',0.7), ('epochs',8)),
    'w180':(('lr',0.7), ('epochs',8))
}

bb_params = {
    'w60':(('lr',0.1), ('mu',1e-5), ('epochs',32)),
    'w120':(('lr',0.1), ('mu',1e-5), ('epochs',32)),
    'w180':(('lr',0.1), ('mu',1e-5), ('epochs',32))
}


bbpool_params = {
     'w60':(('lr',0.01), ('mu',1e-5), ('epochs',32)),
    'w120':(('lr',0.01), ('mu',1e-5), ('epochs',32)),
    'w180':(('lr',0.01), ('mu',1e-5), ('epochs',32))
}


qpt_params = {
    'w60':(('lr',1e-1), ('tau',1e-5), ('get_qpt_mat',get_qpt_mat), ('epochs',24)),
    'w120':(('lr',1e-2), ('tau',1e-5), ('get_qpt_mat',get_qpt_mat), ('epochs',24)),
    'w180':(('lr',1e-2), ('tau',1e-5), ('get_qpt_mat',get_qpt_mat), ('epochs',24))
}

int_params = {
    'w60':(('lr',0.1), ('thr',1e-3), ('damping', 0.1), ('get_mat',get_qpt_mat), ('epochs',5)),
    'w120':(('lr',0.1), ('thr',1e-4), ('damping', 0.1), ('get_mat',get_qpt_mat), ('epochs',25)),
    'w180':(('lr',1e-1), ('thr',1e-3), ('damping', 1e-3), ('get_mat',get_qpt_mat), ('epochs',20))
}

prob2params = {
    'w60':{'weights':weights,'capacity':60, 'n_items':n_items},
    'w120':{'weights':weights,'capacity':120, 'n_items':n_items},
    'w180':{'weights':weights,'capacity':180, 'n_items':n_items}
}

def check_nce_torun(v):
    if  v in (7,8) and torun['MAP']:
        return True 
    elif v in (1,2) and torun['ALL']:
        return True 
    elif v in (4,5) and torun['RELU']:
        return True 
    else:
        return False

def add_estimators(spo_construct, param, **kwarg):
    for prob, cop_params in prob2params.items():
        if spo_construct is not SPOSL:
            tuned = {k:v for k,v in param[prob]}
            yield spo_construct(param=cop_params, suffix=prob, 
                 model=nn.Linear(8,1), **tuned, **kwarg )
        else:
            for v in range(1,10):
                if check_nce_torun(v):
                    lr, e = param[(prob, v)]
                    yield spo_construct(param=cop_params, variant=v,
                    model=nn.Linear(8,1), lr=lr, epochs=e, suffix=prob, **kwarg )



ests = []
for fold in range(args.folds):
    
    if torun['IP']:
        ests += [e for e in add_estimators(Interior, int_params, 
            **fixed_params
        )]

    if torun['QPTL']:
        ests += [e for e in add_estimators(QPTL, qpt_params, 
            **fixed_params
        )]

    if torun['BB']:
        ests += [e for e in add_estimators(BB, bb_params, 
            **fixed_params
        )]

    if torun['SPO']:
        ests += [e for e in add_estimators(SPO, spo_params, 
            **fixed_params
        )]

    if torun['BBP']:
        ests += [e for e in add_estimators(BB_pool, bbpool_params, 
            **fixed_params
        )]

    if torun['SPOP']:
        ests += [e for e in add_estimators(SPO_pool, spopool_params, 
            **fixed_params
        )]

    if torun['2S']:
        ests += [e for e in add_estimators(TwoStage, two_stage_params, 
            criterion=nn.MSELoss , **fixed_params
        )]

    ests += [e for e in add_estimators(SPOSL, nce_params, 
        **fixed_params
    )]


def run(est:SPOSL):
    if args.valid:
        xt = np.vstack((x_train, x_valid))
        yt = np.vstack((y_train, y_valid))
        est.fit(xt, yt)
    else:
        est.fit(x_train, y_train)
    regret = -est.score(x_test, y_test).item()
    scores =  pd.DataFrame([regret], columns=['regret'])
    pred = est.predict(x_test).flatten()
    scores['MSE'] = mse(y_test.flatten(),pred)
    scores['prob'] = est.suffix 
    scores['growth'] = est.growth
    scores['lr'] = est.lr 
    scores['variant'] = est.variant
    scores['model'] = str(est)
    scores['relaxed'] = est.relax
    perf = None
    if args.valid:
        info = est.get_info()
        perf = pd.DataFrame(info['perf'], columns=['regret', 'MSE', 'epoch', 'time'])
        perf['time'] = perf['time'] - perf.at[0, 'time']
        perf['time'] *= 1e-9 
        pool = pd.DataFrame(info['solution_pool'], columns=['size','nunique', 'epoch'])
        perf = pd.merge(perf, pool, on='epoch')
        perf['model'] = str(est)
        del info 
        del pool
    del est
    return scores, perf

launch_jobs(run,ests, args.workers, args.out)



