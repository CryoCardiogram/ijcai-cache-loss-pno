import copy
import torch
import time
import numpy as np
from torch import nn, optim
from nce import NCE_Loss, SPO_NCE_Loss, var2param
from sklearn.metrics import roc_auc_score, mean_squared_error 
from torch.utils.data import Dataset, DataLoader
from Interior.ip_model_whole import IPOfunc
from tqdm.auto import tqdm
from qpthlocal.qp import QPFunction
from qpthlocal.qp import QPSolvers
from qpthlocal.qp import make_gurobi_model


def linear(x,v,**params):
    return x @ v

class SPOSLDataset(Dataset):
    def __init__(self, x,y, params=dict(), solver=None, relaxation=False, sols=None, verbose=False):
        self.x = x
        self.y = y
        if sols is not None:
            self.sols = sols
        else:
            y_iter = range(len(self.y))
            it = tqdm(y_iter) if verbose else y_iter
            self.sols = np.array([solver(self.y[i], relaxation=relaxation, **params) for i in it])
            self.sols = torch.from_numpy(self.sols).float()
        
        self.x = torch.from_numpy(self.x).float()
        self.y = torch.from_numpy(self.y).float()

    
    def __len__(self):
        return len(self.sols)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index], self.sols[index]


class SPOSL:
    """
    SPO with solution pool and NCE-based loss. 

    Args:

        param (dict): named arguments to pass at each solver call

        solver (callable): solves a decision problem to optimality.
            Should have a boolean `relaxation` argument.

        objective_fun (callable): compute the objective value of a decision problem. 
            Fed with ´param´ arguments at each call (default knapsack objective). 

        epoch (int, optional): number of training iterations (default: `20`)

        model (callable): PyTorch neural network

        datawrapper (callable): Dataset constructor. Initialized with x and y during `fit` call.

        unique_only (boolean, optional): set to False to add every solution to the pool, without
            duplicates (default: `True`).

        growth (float, optional): solution pool growth sampling rate (default: `0.0`).

        optimizer (callable): optimizer constructor. Initiliazed with provided 
            :attr:`lr` and :attr:`opt_params`

        relax (boolean, optional): if `True`, train model on LP-relaxation of the problem 
            if available. (default: `False`)

        variant (int, optional): NCE-Pool Loss variant, see `SPO_NCE_Loss` 
            for more info (defaut: `1`). 

        bsizeCOP (int, optional): Mini-batch size (default: `1`).

        suffix (str, optional): name suffix

        maximize (boolean, optional): set to `False` for a minimization problem (default: `True`)

        sol_train (iterable, optional): pre computed solutions for train set

        sol_test (iterabe, optional): pre-computed solutions for test set

    """
    def __init__(self, param=None, solver=None, objective_fun=linear,
     epochs=20, model=None, growth=0.0, optimizer=optim.Adam, relax=False, variant=1, bsizeCOP=1, is_reg=True,
     suffix='', maximize=True, sol_train=None, sol_test=None, datawrapper=SPOSLDataset, scheduler=None,
      validation_idx=None, validation_step=1, early_stop=False, tblog=False,
     lr=1e-3, unique_only=True, save=False, verbose=True, opt_params=dict(), sched_params=dict(), seed=None):
        #self.n_items = n_items
        self.bsizeCOP = bsizeCOP
        self.param = param 
        self.objective_fun = objective_fun
        self.solver = solver
        self.maximize = maximize
        self.variant = variant
        self.datawrapper = datawrapper
        self.scheduler = scheduler
        self.validation_idx = validation_idx
        self.validation_step = validation_step

        self.epochs = epochs
        self.lr = lr
        self.growth = growth
        self._sol_train = sol_train
        self._sol_test = sol_test

        self.optimizer = optimizer
        self.relax = relax
        self.unique_only = unique_only
        self.is_reg = is_reg

        self.model = model
        self.opt_params = opt_params
        self.sched_params = sched_params
        
        self.suffix = suffix
        self.save = save
        self.tblog = tblog
        self.verbose = verbose

        self.seed = seed
        self.early_stop = early_stop
        if seed:
            np.random.seed(seed=seed)
    
    def add_to_pool(self, pool, solution):
        """
        return (pool, unique) 
        
        ´pool´ pool augmented with `solution`

        `unique` boolean assesing solution uniqueness. If self.unique_only is False, unique is always True
        """
        if self.unique_only:
            if (pool == solution).all(1).any():
                return pool, False
            else:
                return torch.cat((solution, pool)), True
                
        else:
            return torch.cat((solution, pool)), True

    def train_fwbw(self, data, init_criterion, init_opti, sol_noispool, epoch=None):
        train_loader = DataLoader(data, batch_size=self.bsizeCOP, shuffle=True)
        optimizer = init_opti
        criterion = init_criterion

        ii = 0
        indices = list(range(len(data)))
        n2solve = int(len(indices) * self.growth)
        growth_indices = set(np.random.choice(indices, size=n2solve, replace=False))
        for xb, yb,solb in train_loader:
            optimizer.zero_grad()
            
            loss = 0
            mbhits = 0
            for i in range(len(yb)): 
                predi = self.model(xb[i])
                if ii in growth_indices:
                    sol = self.solver(predi.detach().squeeze().numpy(), relaxation=self.relax, **self.param)
                    sol = torch.from_numpy(sol).float().unsqueeze(0)
                    self._sol_noispool, added = self.add_to_pool(self._sol_noispool, sol)
                    mbhits += added
                ii+=1
                loss+= criterion(predi, yb[i],solb[i], self._sol_noispool )   

            loss.backward()
            optimizer.step()
            
           
    
    def fit(self, x,y, **fit_params):
        if self.validation_idx is not None:
            x, x_valid = x[:self.validation_idx], x[self.validation_idx:]
            y, y_valid = y[:self.validation_idx], y[self.validation_idx:]
            
        train_dataset = self.datawrapper(x,y, solver=self.solver, params=self.param,
         relaxation=False, sols=self._sol_train, verbose=self.verbose)
    
        optimizer = self.optimizer(self.model.parameters(),lr=self.lr, **self.opt_params)
        if self.scheduler:
            scheduler = self.scheduler(optimizer, **self.sched_params)
        
        # semloss = SPO_NCE_Loss(variant=self.variant, param=self.param, 
        #  linobj=self.objective_fun, maximize=self.maximize)
        semloss = NCE_Loss(self.variant, maximize=self.maximize)
        self._sol_noispool = train_dataset.sols
        self._init_pool = copy.deepcopy(self._sol_noispool)

        e_iter = range(1, self.epochs+1)
        self.__info = {}
        
        for epoch in tqdm(e_iter) if self.verbose else e_iter:
            self.model.train()
            self.train_fwbw(train_dataset, semloss, optimizer, self._sol_noispool, epoch=epoch)
            if self.scheduler:
                scheduler.step(epoch)
            
            if self.validation_idx is not None:
                if epoch % self.validation_step == 0: 
                    start = time.time()
                    regret = -self.score(x_valid, y_valid)
                    pred_score = self.prediction_score(y_valid, self.predict(x_valid))
                    self._valid_t += time.time() - start 
                if self.save:
                        self.__info['perf'].append((regret.item(), pred_score, float(epoch), time.time()- self._valid_t))
                            

    def __str__(self):
        agg, scale = var2param[self.variant]
        return 'L{}{}_e{}_lr{}_g{}_optim{}_{}'.format(
                agg, scale, self.epochs, 
                self.lr, self.growth, self.optimizer.__name__, self.suffix)

    def __repr__(self):
        return str(self)

    def predict(self, x_test ):
        self.model.eval()
        with torch.no_grad():
            if isinstance(x_test, torch.Tensor):
                return self.model(x_test).detach().squeeze().numpy()
            x = self.datawrapper(x_test, x_test , sols=x_test, params=self.param, relaxation=self.relax).x
            pred = self.model(x)
            return pred.detach().squeeze().numpy()

    def get_params(self, deep=True):
        params = {}
        for k,v in vars(self).items():
            if k[0] != '_':
                params[k]=v 
        return params

    def set_params(self, **params):
        for k,v in params.items():
            setattr(self, k,v)
        return self

    def get_info(self):
        return self.__info

    def prediction_score(self, ytrue, ypred):
        scorer = mean_squared_error if self.is_reg else roc_auc_score
        return scorer(ytrue.flatten(), ypred.flatten())

    def score(self, X, y, **score_params):
        test_dataset = self.datawrapper(X,y, params=self.param, solver=self.solver,
         relaxation=False, sols=self._sol_test, verbose=self.verbose)
        test_loader = DataLoader(test_dataset, batch_size=None)
        
        with torch.no_grad():
            
            taskloss = 0
            for i, (xi, yi, sol) in enumerate(test_loader):
                ypred = self.predict(xi)
                solpredi = self.solver(ypred, relaxation=False, **self.param )
    
                regreti = self.maximize * (self.objective_fun(sol, yi, **self.param)
                - self.objective_fun(torch.from_numpy(solpredi).float(), yi,  **self.param))
                taskloss+=regreti
    
        return - taskloss / len(test_dataset)

class TwoStage(SPOSL):
    def __init__(self, criterion=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._criterion = criterion
    def __str__(self):
        return '2stage_e{}_lr{}_{}'.format( self.epochs, 
                self.lr, self.suffix)

    def train_fwbw(self, data, init_criterion, init_opti, sol_noispool, epoch=None):
        train_loader = DataLoader(data, batch_size=self.bsizeCOP, shuffle=True)
        optimizer = init_opti
        criterion = self._criterion()
        ii = 0
        for xb, yb,solb in data:
            optimizer.zero_grad()
            
            loss = 0
            for i in range(len(yb)):
                predi = self.model(xb[i]).squeeze()
                loss+= criterion(predi, yb[i] )   
                ii +=1

            if self.tblog:
                step = ii % (len(train_loader)+1) + (epoch-1)*len(train_loader)
                self._writer.add_scalar('loss', loss.item(),global_step=step )
            
            loss.backward()
            optimizer.step()

class SPO(SPOSL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        return 'SPO_e{}_lr{}_{}'.format( self.epochs, 
                self.lr, self.suffix)

    def train_fwbw_(self, data, init_criterion, init_opti, sol_noispool, epoch=None):
        indices = list(range(len(data)))
        train_loader = DataLoader(data, batch_size=self.bsizeCOP, shuffle=True)
        optimizer = init_opti
        total_loss = 0
        sign = 1 if self.maximize else -1

        ii = 0
        n2solve = int(len(indices) * self.growth)
        growth_indices = set(np.random.choice(indices, size=n2solve, replace=False))
        
        for xb, yb,solb in train_loader:
            optimizer.zero_grad()
            prediction = self.model(xb).squeeze()
            yspo= 2*prediction - yb
            grad_list = []
            for i in range(len(yb)):
                sol_spo = torch.from_numpy(self.solver(yspo[i].detach().numpy(), relaxation=self.relax, **self.param )).float()
                grad_t = (sol_spo - solb[i]) * sign
                grad_list.append(grad_t)
            grad = torch.stack(grad_list,0)
            prediction.retain_grad()
            prediction.backward(gradient=grad)
            optimizer.step()

class SPO_pool(SPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def lookup(self, pool, y):
        _, ind=  self.objective_fun(pool, y, **self.param).max(-1)
        return pool[ind]

    def __str__(self):
        return 'SPOpool_e{}_lr{}_{}'.format( self.epochs, 
                self.lr, self.suffix)

    def train_fwbw(self, data, init_criterion, init_opti, sol_noispool, epoch=None):
        indices = list(range(len(data)))
        train_loader = DataLoader(data, batch_size=self.bsizeCOP, shuffle=True)
        optimizer = init_opti
        total_loss = 0
        sign = 1 if self.maximize else -1
        ii = 0
        n2solve = int(len(indices) * self.growth)
        growth_indices = set(np.random.choice(indices, size=n2solve, replace=False))
        
        for xb, yb,solb in train_loader:
            optimizer.zero_grad()
            prediction = self.model(xb).squeeze()
            yspo=2*prediction - yb
            grad_list = []
            for i in range(len(yb)):
                if ii in growth_indices:
                    sol = self.solver(prediction[i].detach().numpy(), **self.param)
                    sol = torch.from_numpy(sol).float().unsqueeze(0)
                    self._sol_noispool, _ = self.add_to_pool(self._sol_noispool, sol)
                ii+=1
                # look-up sol_spo in pool
                sol_spo = self.lookup(sol_noispool * sign, yspo[i] )
                
                grad_t = (sol_spo - solb[i]) * sign
                grad_list.append(grad_t)
            
            grad = torch.stack(grad_list,0)
            prediction.retain_grad()
            prediction.backward(gradient=grad)
            optimizer.step()

class BB(SPO):
    def __init__(self, mu=1, *args, **kwargs):
        self.mu = mu
        super().__init__(*args, **kwargs)
        

    def __str__(self):
        return 'BB_e{}_lr{}_µ{}_{}'.format( self.epochs, 
                self.lr, self.mu, self.suffix)

    def train_fwbw_(self, data, init_criterion, init_opti, sol_noispool, epoch=None):
        indices = list(range(len(data)))
        train_loader = DataLoader(data, batch_size=self.bsizeCOP, shuffle=True)
        optimizer = init_opti
        sign = 1 if self.maximize else -1

        ii = 0
        n2solve = int(len(indices) * self.growth)
        growth_indices = set(np.random.choice(indices, size=n2solve, replace=False))
        
        for xb, yb,solb in train_loader:
            optimizer.zero_grad()
            y_pred = self.model(xb)
            y_tilde = y_pred + self.mu* yb
            grad_list = []
            for i in range(len(yb)):
                solpred = torch.from_numpy(self.solver(y_pred[i].detach().numpy(), relaxation=self.relax, **self.param)).float()
                soltilde = torch.from_numpy(self.solver(y_tilde[i].detach().numpy(), relaxation=self.relax, **self.param)).float()
                    
                grad_t = -(soltilde - solpred)/self.mu
                grad_list.append(grad_t*sign)

            grad = torch.stack(grad_list,0)
            y_pred.retain_grad()
            y_pred.backward(gradient=grad)
            optimizer.step()

class BB_pool(SPO_pool):
    def __init__(self, mu=1, *args, **kwargs):
        self.mu = mu
        super().__init__(*args, **kwargs)
        

    def __str__(self):
        return 'BBpool_e{}_lr{}_mu{}_{}'.format( self.epochs, 
                self.lr, self.mu, self.suffix)

    def train_fwbw(self, data, init_criterion, init_opti, sol_noispool, epoch=None):
        indices = list(range(len(data)))
        train_loader = DataLoader(data, batch_size=self.bsizeCOP, shuffle=True)
        optimizer = init_opti
        sign = 1 if self.maximize else -1

        ii = 0
        n2solve = int(len(indices) * self.growth)
        growth_indices = set(np.random.choice(indices, size=n2solve, replace=False))
        
        for xb, yb,solb in train_loader:
            optimizer.zero_grad()
            y_pred = self.model(xb).squeeze()
            y_tilde = y_pred + self.mu * yb
            grad_list = []
            for i in range(len(yb)):
                if ii in growth_indices:
                    sol = self.solver(y_pred[i].detach().numpy(), **self.param)
                    sol = torch.from_numpy(sol).float().unsqueeze(0)
                    self._sol_noispool, _ = self.add_to_pool(self._sol_noispool, sol)
                ii+=1
                    
                # look-up sol_spo in pool
                solpred = self.lookup(sol_noispool * sign, y_pred[i])
                soltilde = self.lookup(sol_noispool * sign, y_tilde[i])
                grad_t = -(soltilde - solpred)/self.mu
                grad_list.append(grad_t*sign)

            grad = torch.stack(grad_list,0)
            y_pred.retain_grad()
            y_pred.backward(gradient=grad)
            optimizer.step()


class QPTL(SPOSL):
    def __init__(self, get_qpt_mat=None, tau=1e-4, *args, **kwargs):
        self.tau = tau
        super().__init__(*args, **kwargs)
        self.get_qpt_mat = get_qpt_mat
    
    def __str__(self):
        return 'QPTL_e{}_lr{}_tau{}_{}'.format( self.epochs, 
            self.lr, self.tau, self.suffix)

    def train_fwbw(self, data, init_criterion, init_opti, sol_noispool, epoch=None):
        indices = list(range(len(data)))
        train_dl = DataLoader(data, batch_size=self.bsizeCOP, shuffle=True)
        optimizer = init_opti
        sign = -1 if self.maximize else 1 # QPf minimizes

        ii = 0
        
        A,b, G, h = self.get_qpt_mat(**self.param)
        G_trch = torch.from_numpy(G if G is not None else np.array([])).float()
        h_trch = torch.from_numpy(h if h is not None else np.array([])).float()
        A_trch = torch.from_numpy(A if A is not None else np.array([])).float()
        b_trch = torch.from_numpy(b if b is not None else np.array([])).float()
        Q_trch = (self.tau)*torch.eye(G.shape[1])
        model_params_quad = make_gurobi_model(G, h, 
            A, b, Q_trch.detach().numpy() )

        for xb, yb,solb in train_dl:
            optimizer.zero_grad()

            for i in range(len(yb)):
                y_pred = self.model(xb[i]).squeeze() 
                sol = QPFunction(verbose=False, solver=QPSolvers.GUROBI, 
                        model_params=model_params_quad)(Q_trch.expand(1, *Q_trch.shape),
                         sign*y_pred, G_trch.expand(1, *G_trch.shape), 
                         h_trch.expand(1, *h_trch.shape), 
                         A_trch.expand(1, *A_trch.shape), b_trch.expand(1, *b_trch.shape))
                ii+=1
                loss = -(sol*yb[i]).mean()
                loss.backward() 

            optimizer.step()


class Interior(SPOSL):
    def __init__(self, get_mat=None,method =1,thr = 1e-1, damping=1e-3,max_iter=None,
    *args, **kwargs):
        self.get_mat = get_mat
        self.method = method
        self.thr = thr
        self. damping = damping
        self.max_iter = max_iter
        super().__init__(*args, **kwargs)
        
    
    def __str__(self):
        return 'Interior_e{}_lr{}_thr{}__damping{}_{}'.format( self.epochs, 
            self.lr, self.thr,self.damping, self.suffix)

    def train_fwbw(self, data, init_criterion, init_opti, sol_noispool, epoch=None):
        indices = list(range(len(data)))
        train_dl = DataLoader(data, batch_size=self.bsizeCOP, shuffle=True)
        optimizer = init_opti
        sign = -1 if self.maximize else 1 # QPf minimizes

        ii = 0
        
        A,b, G, h = self.get_qpt_mat(**self.param)
        G_trch = torch.from_numpy(G if G is not None else np.array([])).float()
        h_trch = torch.from_numpy(h if h is not None else np.array([])).float()
        A_trch = torch.from_numpy(A if A is not None else np.array([])).float()
        b_trch = torch.from_numpy(b if b is not None else np.array([])).float()
        Q_trch = (self.tau)*torch.eye(G.shape[1])
        model_params_quad = make_gurobi_model(G, h, 
            A, b, Q_trch.detach().numpy() )

        for xb, yb,solb in train_dl:

            
            optimizer.zero_grad()

            for i in range(len(yb)):
                y_pred = self.model(xb[i]).squeeze() 
                
                _,_, G, h = self.get_mat(mb[i], **self.param)
                G_trch = torch.from_numpy(G).float()
                h_trch = torch.from_numpy(h).float()
                sol = IPOfunc(A=A_trch,b=b_trch,G=G_trch,h=h_trch,
                pc = True,max_iter=self.max_iter, 
                    thr=self.thr,method= self.method,
                    damping= self.damping)(sign*y_pred)
                
                loss = -(sol*yb[i]).mean()
                loss.backward() 

            optimizer.step()