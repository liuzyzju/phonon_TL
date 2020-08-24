import numpy as np 
h_layer1_list = np.random.uniform(110,210,10).astype(int)
h_layer1_list = np.unique(h_layer1_list)
h_layer2_list = np.random.uniform(55,100,5).astype(int)
h_layer2_list = np.unique(h_layer2_list)
h_layer3_list = np.random.uniform(10,50,5).astype(int)
h_layer3_list = np.unique(h_layer3_list)

parameter_list = []
for i in h_layer1_list:
    for j in h_layer2_list:
        for k in h_layer3_list:
            parameter_list.append([i,j,k])
from random import shuffle
shuffle(parameter_list)

parameter_list = np.array(parameter_list)

f = open('parameter_list', 'w')
for i in range(0,len(parameter_list)):
	f.write('{:d} {:d} {:d}\n'.format(parameter_list[i,0],parameter_list[i,1],parameter_list[i,2]))
f.close()