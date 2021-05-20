import torch
import numpy as np
from torch import nn, optim
from copy import deepcopy

var2param = {
        1:('all',0), 2:('all',1), 3:('all',2),
        4:('relu',0), 5:('relu',1), 6:('relu',2),
        7:('map',0), 8:('map',1), 9:('map',2) }




class NCE_Loss(nn.Module):
    r"""
    Implements Contrastive loss over solution cache for SPO.
    
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
    """
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

