#since the graph convolution approach is not working due to N reasons we will try a vanilla conv3s aprroach

import kamitani_data_handler
import numpy as np
import csv
import nibabel as nib
import torch
from torch.utils.data import DataLoader,TensorDataset
import torch.nn.functional as F
from torch.nn import Conv3d,Linear,Module
import torch.optim as optim
from torch.nn import CrossEntropyLoss


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

#get the data
def get_main_data(path,path_csv):
    """
    Function to get main kamitani data
    :param path: path to data
    :return: (flexible)
    """

    kdata = kamitani_data_handler.kamitani_data_handler(path)
    fmri = kdata.get_data()[0]
    fmri = (fmri - fmri.mean()) / fmri.std()

    # fmri_test=kdata.get_data()[2]
    xyz, dim = kdata.get_voxel_loc()
    xyz = np.asarray(xyz).T

    unique_labels, labels_inds = image_labels_from_csv(path_csv)


    return fmri,labels_inds,xyz


def split_data(fmri,labels_inds):
    """
    Split data into train and test in 75-25 split
    :param fmri: fmri data from get_main_data
    :return: train_fmri, train_inds, test_fmri, test_inds
    """
    train_inds = np.array([0, 2, 4, 5, 6, 7])
    test_inds = np.array([1, 3])
    train_fmri = np.zeros([6 * 150, fmri.shape[1]])
    test_fmri = np.zeros([2 * 150, fmri.shape[1]])
    labels_inds_train = np.zeros(6 * 150).astype(int)
    labels_inds_test = np.zeros(2 * 150).astype(int)

    for t in range(0, 150):
        inds = np.arange(0, 6) + t * 6
        train_fmri[inds, :] = fmri[train_inds + 8 * t]
        labels_inds_train[inds] = labels_inds[train_inds + 8 * t]
        print('Training')
        print('Mapping ' + str(inds) + ' to ' + str(train_inds + 8 * t))
        inds = np.arange(0, 2) + t * 2
        test_fmri[inds, :] = fmri[test_inds + 8 * t]
        labels_inds_test[inds] = labels_inds[test_inds + 8 * t]
        print('Testing')
        print('Mapping ' + str(inds) + ' to ' + str(test_inds + 8 * t))
        print('\n')

    return train_fmri, labels_inds_train, test_fmri, labels_inds_test


#get data
fmri,labels_inds,xyz = get_main_data('./data1/Subject3.mat','./data1/imageID_training.csv')
train_fmri, train_inds, test_fmri, test_inds = split_data(fmri,labels_inds)

#make 3d volume
#normalize indices
xyz[:,0] = xyz[:,0] - xyz[:,0].min()
xyz[:,1] = xyz[:,1] - xyz[:,1].min()
xyz[:,2] = xyz[:,2] - xyz[:,2].min()

#world to index
xyz=xyz/3
xyz=xyz.astype(int)

#get sizes
dx = int(xyz[:,0].max() - xyz[:,0].min()) +1
dy = int(xyz[:,1].max() - xyz[:,1].min())+1
dz = int(xyz[:,2].max() - xyz[:,2].min())+1


#make the volumes
train_fmri_3d = np.zeros([train_fmri.shape[0], dx,dy,dz])
test_fmri_3d = np.zeros([test_fmri.shape[0], dx,dy,dz])
mask = np.zeros([dx,dy,dz])

#assign the signal
train_fmri_3d[:,xyz[:,0],xyz[:,1],xyz[:,2]] = train_fmri
test_fmri_3d[:,xyz[:,0],xyz[:,1],xyz[:,2]] = test_fmri
mask[xyz[:,0],xyz[:,1],xyz[:,2]] =1

#test with nii
nii= nib.Nifti1Image(train_fmri_3d[0],np.eye(4))
nib.save(nii,'./data1/test.nii.gz')

train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_fmri_3d.reshape(
    (train_fmri_3d.shape[0],) + (1,) + train_fmri_3d.shape[1:] )),
                                       torch.from_numpy(train_inds.reshape([train_inds.shape[0],1])))

trainloader=DataLoader(train_dataset,batch_size=8,shuffle=True)


class apply_mask(Module):
    def __init__(self,mask):
        self.mask=mask

    def forward(self,x):
        """
        Applies mask
        :param x: input of size [B, Cin, x, y, z]
        :return: flattened masked vector
        """

        return x[:,mask==1,:]

last=16
class net(Module):
    def __init__(self,mask):
        super(net, self).__init__()
        self.mask = mask
        self.N_mask=(mask==1).sum().astype(int)

        self.conv1 = Conv3d(1, 8, 3,padding=1)
        self.conv2 = Conv3d(8, 16, 3,padding=1)
        self.conv3 = Conv3d(16, 16, 3,padding=1)
        #self.conv4 = Conv3d(8, 8, 3,padding=1)
        self.FC1 = Linear(last*self.N_mask,150) #150 is the categories

    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        # x = self.conv4(x)
        # x = F.relu(x)
        x = x[:,:,self.mask==1]
        x=x.view(-1,last*self.N_mask)
        x=self.FC1(x)

        return(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = net(mask).to(device)
model = model.double()

criterion = CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.005)


batch_print=50
for epoch in range(100):
    running_loss= 0.0
    model.train()
    for i,train_data in enumerate(trainloader):
        input=train_data[0]
        target = F.one_hot(train_data[1][:,0],150).float()

        optimizer.zero_grad()
        outputs=model(input)

        loss = criterion(outputs,target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        #print(running_loss)
        if i % batch_print == batch_print-1:  # print every batch_print mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / batch_print:.7f}')
            running_loss = 0.0

            model.eval()
            test=model(torch.from_numpy(test_fmri_3d.reshape((test_fmri_3d.shape[0],)+(1,)+test_fmri_3d.shape[1:])))
            correct = torch.eq(torch.max(F.softmax(test), dim=1)[1],
                                torch.from_numpy(test_inds)).view(-1)
            print(torch.sum(correct))