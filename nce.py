import torch
import numpy as np
from torch import nn, optim
from copy import deepcopy

var2param = {
        1:('all',0), 2:('all',1), 3:('all',2),
        4:('relu',0), 5:('relu',1), 6:('relu',2),
        7:('map',0), 8:('map',1), 9:('map',2) }

class SPO_NCE_Loss(nn.Module):
    """
    Implements Contrastive loss over solution pool for SPO.
    
    Parameters:
        variant: variant for the loss [1-9] 
            1) NCE (ĉ)
            2) NCE (ĉ-c)
            3) NCE (2ĉ-c)
            4) ReLU (ĉ)
            5) ReLU (ĉ-c)
            6) ReLU (2ĉ-c)
            7) MAP (ĉ)
            8) MAP (ĉ-c)
            9) MAP (2ĉ-c)

        linobj: function(decision_var, prediction, **kwargs) returns objective_value

        param: Combinatorial problem parameters. 
    """
    def __init__(self, variant:int, param=dict(), linobj=lambda x,v: x @ v, maximize=True):
        super(SPO_NCE_Loss, self).__init__()
        self.relu = nn.ReLU()
        self.aggregation = {
            'relu':lambda x: torch.sum(self.relu(x)),
            'map':lambda x: torch.max(x),
            'all': lambda x: torch.sum(x)
        }
        agg, scale = var2param[variant]

        self.aggtype = agg
        self.scaletrick = scale
        self.step = 0
        self.diff_vec_list = []
        self.linobj = linobj
        self.param = param
        self.max_ind = 1 if maximize else -1
        
    def forward(self, y_pred, y_true, sol_true, sol_spo):
        sol_diff = (sol_spo - sol_true ) * self.max_ind
        yhat = y_pred if self.scaletrick == 0 else self.scaletrick * y_pred - y_true
        # (v* - v_i) . pred 
        # (batched dot product for each solution v_i)
        objectve_value_diff = self.linobj(sol_diff, yhat, **self.param)
        
        aggfunc = self.aggregation[self.aggtype]
        loss = aggfunc(objectve_value_diff)
        return loss

class NCE_Loss(nn.Module):
    def __init__(self, variant, maximize=True):
        super(NCE_Loss, self).__init__()
        self.variants = variant
        self.max_ind = 1 if maximize else -1
    
    def forward(self, y_pred_i, y_train_i, sol_train_i, batch_sol_spos_torch):
        y_pred_i = y_pred_i.view(*y_train_i.shape)
        if self.variants==1:#1
            loss = ((self.max_ind*(batch_sol_spos_torch - sol_train_i)*y_pred_i).sum())
        if self.variants==3: #1b
            loss = ((self.max_ind*(batch_sol_spos_torch - sol_train_i)*(2*y_pred_i - y_train_i)).sum())
        if self.variants==2:#1a
            loss = ((self.max_ind*(batch_sol_spos_torch - sol_train_i)*(y_pred_i - y_train_i)).sum())
        if self.variants==5: #2a
            m = torch.nn.ReLU()
            loss = m((self.max_ind*(batch_sol_spos_torch - sol_train_i)*(y_pred_i - y_train_i)).sum(dim=1)).sum()
        if self.variants==10:
            m = torch.nn.ReLU()
            loss = m((self.max_ind*(batch_sol_spos_torch - sol_train_i)*(  y_train_i -y_pred_i)).sum(dim=1)).sum()
        if self.variants==4: #2
            m = torch.nn.ReLU()
            loss = m((self.max_ind*(batch_sol_spos_torch- sol_train_i)*y_pred_i).sum(dim=1)).sum()    
        if self.variants==7: #3
            loss =  (self.max_ind*(batch_sol_spos_torch - sol_train_i)*y_pred_i).sum(dim=1).max()                    
        if self.variants==8: #3a
            loss = (self.max_ind*(batch_sol_spos_torch - sol_train_i)*(y_pred_i - 
                y_train_i)).sum(dim=1).max() 
        if self.variants==9: #3b
            loss = (self.max_ind*(batch_sol_spos_torch - sol_train_i)*(2*y_pred_i - 
                y_train_i)).sum(dim=1).max()
        if self.variants==6: #2b
            m = torch.nn.ReLU()
            loss = m((self.max_ind*(batch_sol_spos_torch - sol_train_i)*(2*y_pred_i - y_train_i)).sum(dim=1)).sum()
        return loss


if __name__ == '__main__':
    # semloss = SPO_Semantic_Loss( aggtype='max', linobj=lambda x,v:x@v)
    # semloss_fix = NCE_Loss(7)
    n_item = 4
    batch_sol_size = 1
    pred = torch.randn(n_item) * 5000
    # sol_true = torch.from_numpy(np.random.randint(0,2, n_item)).float()
    # sol_true[0] = 1
    sol_true = torch.from_numpy(np.array([1,1,0,0])).float()
    # sol_spos = torch.from_numpy(np.random.randint(0,2,(batch_sol_size, n_item))).float()
    # sol_spos[0,-1]=1
    sol_spos = torch.from_numpy(np.array([1,1,0,1])).float()
    sol_spos = torch.stack((sol_spos, sol_true))
    y_true = (sol_true + 5) ** 3

    var2param = {
        1:('all',0), 2:('all',1), 3:('all',2),
        4:('relu',0), 5:('relu',1), 6:('relu',2),
        7:('map',0), 8:('map',1), 9:('map',2) }

    for variant in range(1,10):
        agg, reg = var2param[variant]
        print(agg, reg)
        semloss = SPO_NCE_Loss( variant, maximize=True)
        semloss_fix = NCE_Loss(variant, maximize=True)
        loss_fix = semloss_fix(pred,y_true, sol_true.float(), sol_spos.float())
        loss_old = semloss(pred,y_true, sol_true.float(), sol_spos.float())
        print(loss_fix, loss_old)
        if loss_fix - loss_old != 0:
            print('variant: {}, scale: {}'.format(agg, reg))
            print('old: ', loss_old.item())
            # sol_diff = sol_spos - sol_true
            # obj_diff =  sol_diff @ pred
            # print(max(obj_diff))
            print('fix: ', loss_fix.item())
