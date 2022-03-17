import numpy as np
from sklearn.neighbors import NearestNeighbors
import kamitani_data_handler
#import bdpy
import matplotlib.pyplot as plt
from sklearn.manifold import spectral_embedding
import csv
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

def adjency_from_xyz(xyz):
    """
    Returns an adjacency matrix from coordinates
    :param xyz: coordinates array with shape N,3
    :return: A (adjancency matrix), dist (k means),inds (k means)
    """
    nbrs = NearestNeighbors(n_neighbors=10).fit(xyz)
    dist, inds = nbrs.kneighbors(xyz)
    A=np.zeros([xyz.shape[0],xyz.shape[0]])
    #loop over nodes, if no non-diagonal neighbour use one diagonal neighbour
    non_diagonal_nodes=0
    diagonal_nodes=0
    #there is some stuff in this loop regarding diagonal vs. non-diagnonal connections
    for i, d in enumerate(dist):
        connected_nodes=np.where(d==3)[0]
        if connected_nodes.shape[0] >0:
            A[i,inds[i,connected_nodes]]=1
            non_diagonal_nodes+=1
            print(non_diagonal_nodes)
        else:
            print('no non-diagonal connections found, looking for diagonal ones')
            eps=1/1000
            diag_dist=np.sqrt(3**2+3**2)+eps
            connected_nodes = np.where(d <= diag_dist & d>0)[0]
            if connected_nodes.shape[0] == 0:
                print('no diagonal connections')
                raise ValueError("no shortest diagonal connection")
            A[i,inds[i,connected_nodes]]=1
            diagonal_nodes+=1
            print(diagonal_nodes)
    #a,b=np.where(dist==3)
    #A[a,inds[a,b]]=1
    return A,dist,inds

def image_labels_from_csv(csvpath):
    """
    Assigns an integer to each unique label
    :param csvpath:
    :return: unique_labels, labels_inds
    """
    labels = []
    with open(csvpath) as csvfile:
        rdr = csv.reader(csvfile, delimiter=',')
        for r in rdr:
            #print(r[0].split('.')[0])
            labels.append(r[0].split('.')[0])
    labels = np.asarray(labels)
    unique_labels,labels_inds= np.unique(labels,return_inverse=1)
    return unique_labels,labels_inds

def edge_index_from_adj(A):
    """
    Converts to COO format
    :param A: adjacency matrix
    :return: [row,col]
    """
    row,col = (A>0).nonzero()
    return np.asarray([row,col])

unique_labels,labels_inds= image_labels_from_csv('./data1/imageID_training.csv')

kdata=kamitani_data_handler.kamitani_data_handler('./data1/Subject3.mat')
# #imgs=np.load('./data/images/images_112/train_images.npy')
#
fmri=kdata.get_data()[0]
xyz,dim=kdata.get_voxel_loc()
xyz=np.asarray(xyz).T
#
A,dist,inds=adjency_from_xyz(xyz)

edge_index=torch.from_numpy(edge_index_from_adj(A))

data_list=[]
for i,this_fmri in enumerate(fmri):
    data_list.append(Data(x=torch.from_numpy(this_fmri.reshape([edge_index.max()+1,1])),
                          edge_index=edge_index,
                          num_nodes=torch.tensor(edge_index.max()+1),
                                                 y=torch.tensor(labels_inds[i])))


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x#F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
model = model.double()

#graph= Data(x=fmri[0],edge_index=edge_index,num_nodes=edge_index.max()+1)

# vis = to_networkx(graph)

#node_labels = graph.y.numpy()

# import matplotlib.pyplot as plt
# plt.figure(1,figsize=(15,13))
# nx.draw(vis, cmap=plt.get_cmap('Set3'),node_size=70,linewidths=6)
# plt.show()
#
# #
# embedding = spectral_embedding(A,n_components=50)
#
#
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
#
# for i in range(0,embedding.shape[1]):
#     ax.view_init(elev=71, azim=27)
#     ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=embedding[:,i])
#     plt.savefig('./data/snaps/harmonics' + str(i) + '.png')