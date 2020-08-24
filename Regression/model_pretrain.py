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


df = pd.read_pickle('ph_gap_new.pkl')
desc_hand = pd.read_pickle('desc.pkl')
prop = df['bandgap']

desc_ph = desc_hand[df['phonon bandgap']>0]
prop_ph = df[df['phonon bandgap']>0]['phonon bandgap']


scaler = preprocessing.MinMaxScaler()
desc_el_scaler = scaler.fit_transform(desc_hand)
desc_ph_scaler = scaler.transform(desc_ph)


parameter_list = np.loadtxt('parameter_list')

device = 'cuda'

X = torch.from_numpy(np.array(desc_el_scaler)).type(torch.cuda.FloatTensor)
Y = torch.from_numpy(np.array(prop)).type(torch.cuda.FloatTensor)
Y = Y.reshape([len(Y),1])

X = X.to(device)
Y = Y.to(device)


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

feature_size = 290
out_layer = 1
LR = 0.01
BATCH_SIZE = 200


for i in range(len(parameter_list)):
	h_layer1 = np.int(parameter_list[i][0])
	h_layer2 = np.int(parameter_list[i][1])
	h_layer3 = np.int(parameter_list[i][2])
	cwd = os.getcwd()
	path = cwd+'/pre_train_folder_'+str(h_layer1)+'_'+str(h_layer2)+'_'+str(h_layer3)+'_0.1'
	os.mkdir(path)

	torch_data  = torch.utils.data.TensorDataset(X, Y)
	loader = DataLoader(dataset=torch_data, batch_size=BATCH_SIZE, shuffle=True)
	EPOCH = 300
	test_net = TransNet()
	test_net = test_net.to(device)
	opt_adam = torch.optim.Adam(test_net.parameters(), lr=LR, weight_decay=0.0001)
	loss_func = nn.MSELoss()
	loss_list = []
	for epoch in range(EPOCH):
		for step, (b_x, b_y) in enumerate(loader):
			b_x = b_x.float()
			b_y = b_y.float()
			pre = test_net(b_x)
			loss = loss_func(pre, b_y)
			opt_adam.zero_grad()
			loss.backward()
			opt_adam.step()
			loss_list.append(loss)
	# finished training
	torch.save(test_net,path+'/'+str(h_layer1)+'_'+str(h_layer2)+'_'+str(h_layer3)+'_model')
