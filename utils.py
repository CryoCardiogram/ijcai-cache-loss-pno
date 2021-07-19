import numpy as np 
import pandas as pd 
import glob
import itertools
import os 
from tqdm.auto import tqdm
from concurrent.futures import  ThreadPoolExecutor, ProcessPoolExecutor
import concurrent
import traceback


def iterate_values(S):
    keys, values = zip(*S.items())
    L =[]

    for row in itertools.product(*values):
        L.append( dict(zip(keys, row)))
    return L  

def custom_cvfolds(n_train, n_test, n_fold=5):
    i = 0
    while i < n_fold:
        idx1 = np.arange(0, n_train , dtype=int)
        idx2 = np.arange(n_train, n_train+n_test, dtype=int)
        yield idx1, idx2
        i += 1




def launch_jobs(run, ests, n_worker, out):
    list_res = []
    
    with tqdm(total=len(ests)) as pbar, ProcessPoolExecutor(max_workers=n_worker) as executor:
        futures = [executor.submit(run, e) for e in reversed(ests)]
        for f in futures:
            f.add_done_callback(lambda f:pbar.update())
        for future in concurrent.futures.as_completed(futures):
            try:
                list_res.append(future.result())
            except BaseException as e:
                print(traceback.print_tb(e))
                print('fail to run:', repr(e))
                
            if len(list_res) > 10:
                scores_l, perf_l = zip(*list_res)
                list_res = []
                scores = pd.concat(scores_l)
                with open(out, 'a') as f:
                    scores.to_csv(f,index=False, header=f.tell()==0)
                del scores
                print(perf_l)
                if not any(e is None for e in perf_l):
                    perfs = pd.concat(perf_l)
                    with open(out.split('.')[0] + '_perf.csv', 'a') as f2:
                        perfs.to_csv(f2, index=False, header=f2.tell()==0)
                    del perfs

    scores_l, perf_l = zip(*list_res)
    list_res = []
    scores = pd.concat(scores_l)
    with open(out, 'a') as f:
        scores.to_csv(f,index=False, header=f.tell()==0)
    del scores
    if not any(e is None for e in perf_l):
        perfs = pd.concat(perf_l)
        with open(out.split('.')[0] + '_perf.csv', 'a') as f2:
            perfs.to_csv(f2, index=False, header=f2.tell()==0)
        del perfs