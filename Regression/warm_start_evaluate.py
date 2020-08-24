import numpy as np 
import pandas as pd
from torch.utils.data import DataLoader
import torch
import os
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn import preprocessing
from skorch.regressor import NeuralNetRegressor
from sklearn.neural_network import MLPRegressor

class TransNet(nn.Module):
    def __init__(self):
      super(TransNet, self).__init__()
      self.sharedlayer = nn.Sequential(
        nn.Linear(feature_size, h_layer1),
        nn.BatchNorm1d(h_layer1),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(h_layer1, h_layer2),
        nn.BatchNorm1d(h_layer2),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(h_layer2, h_layer3),
        nn.BatchNorm1d(h_layer3),
        nn.ReLU(),
        nn.Dropout(0.1),
      )
      self.finallayer = nn.Sequential(
        nn.Linear(h_layer3, out_layer)
      )
    def forward(self, x):
      x = self.sharedlayer(x)
      h_shared  = self.finallayer(x)
      return h_shared
df = pd.read_pickle('ph_cv.pkl')
desc_hand = pd.read_pickle('desc.pkl')

desc_ph = desc_hand[df['phonon bandgap']>0]
prop_ph = df[df['phonon bandgap']>0]['phonon bandgap']
prop_ph = np.array(prop_ph)

scaler = preprocessing.MinMaxScaler()
desc_el_scaler = scaler.fit_transform(desc_hand)
desc_ph_scaler = scaler.transform(desc_ph)

device = 'cuda'
LR = 0.01
MAX_EPOCHS = 500

out_layer = 1

parameter_list = np.loadtxt('parameter_list')
cwd = os.getcwd()

BATCH_SIZE = len(desc_ph)

for i in range(0,len(parameter_list)):
    h_layer1 = np.int(parameter_list[i][0])
    h_layer2 = np.int(parameter_list[i][1])
    h_layer3 = np.int(parameter_list[i][2])
    path = cwd+'/pre_train_folder_'+str(h_layer1)+'_'+str(h_layer2)+'_'+str(h_layer3)+'_0.1'  
    model = torch.load(path+'/'+str(h_layer1)+'_'+str(h_layer2)+'_'+str(h_layer3)+'_model')
    feature_size = model.sharedlayer[0].in_features  
    class PretrainedModel(nn.Module):
        def __init__(self):
            super(PretrainedModel, self).__init__()
            self.sharedlayer = nn.Sequential(
                nn.Linear(feature_size, h_layer1),
                nn.BatchNorm1d(h_layer1),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(h_layer1, h_layer2),
                nn.BatchNorm1d(h_layer2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(h_layer2, h_layer3),
                nn.BatchNorm1d(h_layer3),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            self.sharedlayer[0].weight = torch.nn.Parameter(model.sharedlayer[0].weight)
            #self.sharedlayer[0].weight.requires_grad = False
            self.sharedlayer[0].bias = torch.nn.Parameter(model.sharedlayer[0].bias)
            #self.sharedlayer[0].bias.requires_grad = False
            self.sharedlayer[4].weight = torch.nn.Parameter(model.sharedlayer[4].weight)
            #self.sharedlayer[4].weight.requires_grad = False
            self.sharedlayer[4].bias = torch.nn.Parameter(model.sharedlayer[4].bias)
            #self.sharedlayer[4].bias.requires_grad = False
            self.sharedlayer[8].weight = torch.nn.Parameter(model.sharedlayer[8].weight)
            #self.sharedlayer[8].weight.requires_grad = False
            self.sharedlayer[8].bias = torch.nn.Parameter(model.sharedlayer[8].bias)
            #self.sharedlayer[8].bias.requires_grad = False
        def forward(self, x):
            x = self.sharedlayer(x)
            h_shared  = self.finallayer(x)
            return h_shared

    net_regr = NeuralNetRegressor(
        PretrainedModel,
        max_epochs=MAX_EPOCHS,
        optimizer = torch.optim.Adam,
        optimizer__weight_decay=0.0001,
        batch_size = BATCH_SIZE,
        lr = LR,
        train_split = None,
        device='cuda',
    )
    X_regr = desc_ph_scaler.astype(np.float32)
    y_regr = prop_ph.astype(np.float32) 
    y_regr = y_regr.reshape(-1, 1)
    f = open(path+'/bandgap_transfer_performance_6','w')

    for j in range(3): 
        net_regr = NeuralNetRegressor(
            PretrainedModel,
            max_epochs=MAX_EPOCHS,
            optimizer = torch.optim.Adam,
            optimizer__weight_decay=0.0001,
            batch_size = BATCH_SIZE,
            lr = LR,
            train_split = None,
            device='cuda',
            )
        net_regr.fit(X_regr, y_regr)
        for k in range(5): 
            predicted = cross_val_predict(net_regr, X_regr, y_regr, cv=5)
            mae = mean_absolute_error(y_regr, predicted)
            r2 = r2_score(y_regr, predicted)
            f.write('{:03d} {:03d} {:.8f} {:.8f}\n'.format(j, k, mae, r2))
    f.close()
    
