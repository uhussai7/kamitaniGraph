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
from torch_geometric.nn import GCNConv, Linear
import torch.nn.functional as F
#from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader,TensorDataset
import torch.optim as optim
from torch.nn import CrossEntropyLoss

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

#getting labels
unique_labels,labels_inds= image_labels_from_csv('./data1/imageID_training.csv')
#unique_labels_test,labels_inds_test= image_labels_from_csv('./data1/imageID_test.csv')



#getting and extracting fmri and postion data
kdata=kamitani_data_handler.kamitani_data_handler('./data1/Subject3.mat')
fmri=kdata.get_data()[0]
fmri=(fmri-fmri.mean())/fmri.std()
#fmri_test=kdata.get_data()[2]
xyz,dim=kdata.get_voxel_loc()
xyz=np.asarray(xyz).T

#for pooling we need to figure out which ROIs overlap
ROI_meta_inds=np.arange(8,18)
overlap_matrix=np.zeros([ROI_meta_inds.shape[0],ROI_meta_inds.shape[0]])
for rr1,r1 in enumerate(ROI_meta_inds):
    for rr2,r2 in enumerate(ROI_meta_inds):
        occ_vox_1= kdata.voxel_meta[r1] #occupied voxels
        occ_vox_2= kdata.voxel_meta[r2]
        overlap_matrix[rr1,rr2] = ((occ_vox_1 + occ_vox_2[r2])==2).sum()/(occ_vox_1==1).sum()
        overlap_matrix[rr2,rr1]=overlap_matrix[rr1,rr2]
clusters = torch.from_numpy(kdata.voxel_meta[ROI_meta_inds].astype(int))

#split provided training data into new training and test dataset
#accroding to a 75-25 split we have 6-2 split of the eight categories
train_inds=np.array([0,2,4,5,6,7])
test_inds=np.array([1,3])
train_fmri=np.zeros([6*150,fmri.shape[1]])
test_fmri=np.zeros([2*150,fmri.shape[1]])
labels_inds_train=np.zeros(6*150).astype(int)
labels_inds_test=np.zeros(2*150).astype(int)

for t in range(0,150):
    inds=np.arange(0,6)+t*6
    train_fmri[inds,:]=fmri[train_inds+8*t]
    labels_inds_train[inds]=labels_inds[train_inds+8*t]
    print('Training')
    print('Mapping '+str(inds)+ ' to '+str(train_inds+8*t))
    inds=np.arange(0,2)+t*2
    test_fmri[inds,:]=fmri[test_inds+8*t]
    labels_inds_test[inds]=labels_inds[test_inds+8*t]
    print('Testing')
    print('Mapping ' + str(inds) + ' to ' + str(test_inds + 8 * t))
    print('\n')

#make adjacency matrix and edge_index
A,dist,inds=adjency_from_xyz(xyz)
edge_index=torch.from_numpy(edge_index_from_adj(A))

#OPTION 1: use data_list for all the signals and graphs (inefficient since graph is same)
# data_list=[]
# for i,this_fmri in enumerate(fmri):
#     data_list.append(Data(x=torch.from_numpy(this_fmri.reshape([edge_index.max()+1,1])),
#                           edge_index=edge_index,
#                           num_nodes=torch.tensor(edge_index.max()+1),
#                                                 y=torch.tensor(labels_inds[i])))
# trainloader=DataLoader(data_list,batch_size=8,shuffle=True)

#OPTION 2: use pytorch dataloader only on signal and labels
train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_fmri.reshape(train_fmri.shape + (1,))),
                                       torch.from_numpy(labels_inds_train.reshape([labels_inds_train.shape[0],1])))
trainloader=DataLoader(train_dataset,batch_size=8,shuffle=True)



class roi_pool(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x,clusters):
        #clusters should have shape [N_clusters,nodes]
        N_batch = x.shape[0]
        N_clusters=clusters.shape[0]
        Cin = x.shape[-1]
        out= torch.zeros([N_batch,N_clusters,Cin]).double()
        for c in range(0,N_clusters):
            #print('computing max in clusters')
            out[:,c,:]=x[:,clusters[c]==1,:].mean(dim=1)[0]
        return out

last = 4
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 4,node_dim=1)
        self.conv2 = GCNConv(4, 4, node_dim=1)
        self.conv3 = GCNConv(4, 4, node_dim=1)
        self.conv4 = GCNConv(4, last,node_dim=1)
        self.FC1 = Linear(fmri.shape[1]*last, 1000)
        self.FC2 = Linear(1000,500)
        self.FC3 = Linear(500,250)
        self.pool = roi_pool()
        #self.FC4 = Linear(clusters.shape[0] * last, labels_inds.max() + 1)
        #self.FC4 = Linear(fmri.shape[1]*last,labels_inds.max()+1)
        self.FC4 = Linear(250,labels_inds.max()+1)

    # def forward(self, data):
    #     x, edge_index = data.x, data.edge_index
    def forward(self, x,edge_index):
        #x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x=F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        #x = self.pool(x,clusters)

        x=x.view(-1,fmri.shape[1]*last)
        #x = x.view(-1, clusters.shape[0] * last)
        x=self.FC1(x)
        x = F.relu(x)
        x=self.FC2(x)
        x = F.relu(x)
        x=self.FC3(x)
        x = F.relu(x)
        x = self.FC4(x)

        return x#F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
model = model.double()

criterion = CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01)


batch_print=50
for epoch in range(100):
    running_loss= 0.0
    model.train()
    for i,train_data in enumerate(trainloader):
        input=train_data[0]
        target = F.one_hot(train_data[1][:,0],150).float()

        optimizer.zero_grad()
        outputs=model(input,edge_index)

        loss = criterion(outputs,target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()


        #print(running_loss)
        if i % batch_print == batch_print-1:  # print every batch_print mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / batch_print:.7f}')
            running_loss = 0.0

            model.eval()
            test=model(torch.from_numpy(test_fmri.reshape(test_fmri.shape + (1,))),edge_index)
            correct = torch.eq(torch.max(F.softmax(test), dim=1)[1],
                               torch.from_numpy(labels_inds_test)).view(-1)
            print(torch.sum(correct))
        #if i==1: break

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