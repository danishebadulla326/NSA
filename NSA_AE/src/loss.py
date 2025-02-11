#!/usr/bin/python3
import numpy as np
import torch
import torch.nn as nn
import ripserplusplus as rpp_py
# from gph.python import ripser_parallel

def lp_loss(a, b, p=2):
    return (torch.sum(torch.abs(a-b)**p))

def get_indicies(DX, rc, dim, card):
    dgm = rc['dgms'][dim]
    pairs = rc['pairs'][dim]

    # Retrieve vertices v_a and v_b by picking the ones achieving the maximal
    # distance among all pairwise distances between the simplex vertices
    indices, pers = [], []
    for i in range(len(pairs)):
        s1, s2 = pairs[i]
        if len(s1) == dim+1 and len(s2) > 0:
            l1, l2 = np.array(s1), np.array(s2)
            i1 = [s1[v] for v in np.unravel_index(np.argmax(DX[l1,:][:,l1]),[len(s1), len(s1)])]
            i2 = [s2[v] for v in np.unravel_index(np.argmax(DX[l2,:][:,l2]),[len(s2), len(s2)])]
            indices += i1
            indices += i2
            pers.append(dgm[i][1] - dgm[i][0])
    
    # Sort points with distance-to-diagonal
    perm = np.argsort(pers)
    indices = list(np.reshape(indices, [-1,4])[perm][::-1,:].flatten())
    
    # Output indices
    indices = indices[:4*card] + [0 for _ in range(0,max(0,4*card-len(indices)))]
    return list(np.array(indices, dtype=np.compat.long))

def Rips(DX, dim, card, n_threads, engine):
    # Parameters: DX (distance matrix), 
    #             dim (homological dimension), 
    #             card (number of persistence diagram points, sorted by distance-to-diagonal)
    if dim < 1:
        dim = 1
        
    if engine == 'ripser':
        DX_ = DX.numpy()
        DX_ = (DX_ + DX_.T) / 2.0 # make it symmetrical
        DX_ -= np.diag(np.diag(DX_))
        rc = rpp_py.run("--format distance --dim " + str(dim), DX_)
    elif engine == 'giotto':
        rc = ripser_parallel(DX, maxdim=dim, metric="precomputed", collapse_edges=False, n_threads=n_threads)
    
    all_indicies = [] # for every dimension
    for d in range(1, dim+1):
        all_indicies.append(get_indicies(DX, rc, d, card))
    return all_indicies

class RTD_differentiable(nn.Module):
    def __init__(self, dim=1, card=50, mode='minimum', n_threads=25, engine='giotto'):
        super().__init__()
            
        if dim < 1:
            raise ValueError(f"Dimension should be greater than 1. Provided dimension: {dim}")
        self.dim = dim
        self.mode = mode
        self.card = card
        self.n_threads = n_threads
        self.engine = engine
        
    def forward(self, Dr1, Dr2, immovable=None):
        # inputs are distance matricies
        d, c = self.dim, self.card
        
        if Dr1.shape[0] != Dr2.shape[0]:
            raise ValueError(f"Point clouds must have same size. Size Dr1: {Dr1.shape} and size Dr2: {Dr2.shape}")
            
        if Dr1.device != Dr2.device:
            raise ValueError(f"Point clouds must be on the same devices. Device Dr1: {Dr1.device} and device Dr2: {Dr2.device}")
            
        device = Dr1.device
        # Compute distance matrices
#         Dr1 = torch.cdist(r1, r1)
#         Dr2 = torch.cdist(r2, r2)

        Dzz = torch.zeros((len(Dr1), len(Dr1)), device=device)
        if self.mode == 'minimum':
            Dr12 = torch.minimum(Dr1, Dr2)
            DX = torch.cat((torch.cat((Dzz, Dr1.T), 1), torch.cat((Dr1, Dr12), 1)), 0)
            if immovable == 2:
                DX_2 = torch.cat((torch.cat((Dzz, Dr1.T), 1), torch.cat((Dr1, Dr1), 1)), 0)   # Transfer gradient for edge minimization to edges in cloud #1
            elif immovable == 1:
                DX_2 = torch.cat((torch.cat((Dzz, Dr1.T), 1), torch.cat((Dr1, Dr2), 1)), 0)   # Transfer gradient from edge minimization to edges in cloud #2
            else:
                DX_2 = DX
        else:
            Dr12 = torch.maximum(Dr1, Dr2)
            DX = torch.cat((torch.cat((Dzz, Dr12.T), 1), torch.cat((Dr12, Dr2), 1)), 0)
            if immovable == 2:
                DX_2 = torch.cat((torch.cat((Dzz, Dr1.T), 1), torch.cat((Dr1, Dr2), 1)), 0)   # Transfer gradient for edge minimization to edges in cloud #1
            elif immovable == 1:
                DX_2 = torch.cat((torch.cat((Dzz, Dr2.T), 1), torch.cat((Dr2, Dr2), 1)), 0)   # Transfer gradient from edge minimization to edges in cloud #2
            else:
                DX_2 = DX
        
        # Compute vertices associated to positive and negative simplices 
        # Don't compute gradient for this operation
        all_ids = Rips(DX.detach().cpu(), self.dim, self.card, self.n_threads, self.engine)
        all_dgms = []
        for ids in all_ids:
            # Get persistence diagram by simply picking the corresponding entries in the distance matrix
            tmp_idx = np.reshape(ids, [2*c,2])
            if self.mode == 'minimum':
                dgm = torch.hstack([torch.reshape(DX[tmp_idx[::2, 0], tmp_idx[::2, 1]], [c,1]), torch.reshape(DX_2[tmp_idx[1::2, 0], tmp_idx[1::2, 1]], [c,1])])
            else:
                dgm = torch.hstack([torch.reshape(DX_2[tmp_idx[::2, 0], tmp_idx[::2, 1]], [c,1]), torch.reshape(DX[tmp_idx[1::2, 0], tmp_idx[1::2, 1]], [c,1])])
            all_dgms.append(dgm)
        return all_dgms
    
class RTDLoss(nn.Module):
    def __init__(self, dim=1, card=50, n_threads=25, engine='giotto', mode='minimum', is_sym=True, lp=1.0, **kwargs):
        super().__init__()

        self.is_sym = is_sym
        self.mode = mode
        self.p = lp
        self.rtd = RTD_differentiable(dim, card, mode, n_threads, engine)
    
    def forward(self, x_dist, z_dist):
        # x_dist is the precomputed distance matrix
        # z is the batch of latent representations
        loss = 0.0
        loss_xz = 0.0
        loss_zx = 0.0
        rtd_xz = self.rtd(x_dist, z_dist, immovable=1)
        if self.is_sym:
            rtd_zx = self.rtd(z_dist, x_dist, immovable=2)
        for d, rtd in enumerate(rtd_xz): # different dimensions
            loss_xz += lp_loss(rtd_xz[d][:, 1], rtd_xz[d][:, 0], p=self.p)
            if self.is_sym:
                loss_zx += lp_loss(rtd_zx[d][:, 1], rtd_zx[d][:, 0], p=self.p)
        loss = (loss_xz + loss_zx) / 2.0
        return loss_xz, loss_zx, loss

class NSALoss(nn.Module):
    def __init__(self, mode='raw', **kwargs):
        super().__init__()
        self.mode = mode
    
    def forward(self, x, z):
        # normA1 = torch.max(torch.sqrt(torch.sum(x**2,axis=1)))
        # normA2 = torch.max(torch.sqrt(torch.sum(z**2,axis=1)))
        if self.mode=='raw':
            normA1 = torch.quantile(torch.sqrt(torch.sum(x**2,axis=1)),0.98)
            normA2 = torch.quantile(torch.sqrt(torch.sum(z**2,axis=1)),0.98)
            
            A1_pairwise = torch.flatten(torch.cdist(x,x))    # compute pairwise dist
            A2_pairwise = torch.flatten(torch.cdist(z,z))    # compute pairwise dist
            A1_pairwise = A1_pairwise/(2*normA1)
            A2_pairwise = A2_pairwise/(2*normA2)
        elif self.mode=='dist':
            A1_pairwise = torch.flatten(x)
            A2_pairwise = torch.flatten(z)

        loss = torch.mean(torch.abs(A2_pairwise - A1_pairwise))
        return loss

class LID_NSALoss_v1(nn.Module):
    def __init__(self, k=5, eps=1e-7, full=False, **kwargs):
        super().__init__()
        self.k = k
        self.eps = eps
        self.full = full
    def compute_neighbor_mask(self, X, normA1):
        #print("Computing N neighbors for ground truth")
        x_dist = torch.cdist(X,X)+self.eps
        #print(x_dist)
        x_dist = x_dist/normA1
        # print(self.k,x_dist.shape)
        if self.full:
            values, indices = torch.topk(x_dist, x_dist.shape[0], largest=False)
        else:
            values, indices = torch.topk(x_dist, self.k+1, largest=False)
        values, indices = values[:,1:], indices[:,1:]
        #values = values/max(values)
        #print(values)
        norm_values=values[:,-1].view(values.shape[0],1)
        #print(norm_values)
        lid_X = -torch.sum(torch.log10(values) - torch.log10(norm_values),axis=1)
        return indices, lid_X
    
    def forward(self, X, Z):
        # normA1 = torch.quantile(torch.sqrt(torch.sum(X**2,axis=1)),0.98)
        # normA2 = torch.quantile(torch.sqrt(torch.sum(Z**2,axis=1)),0.98)
        mean_x = torch.mean(X, dim=0)
        mean_z = torch.mean(Z, dim=0)
        
        normA1 = torch.quantile(torch.sqrt(torch.sum((X - mean_x) ** 2, dim=1)),0.98)
        normA2 = torch.quantile(torch.sqrt(torch.sum((Z - mean_z) ** 2, dim=1)),0.98)

        nn_mask, lid_X = self.compute_neighbor_mask(X, normA1)
        z_dist = torch.cdist(Z,Z)+self.eps
        z_dist = z_dist/normA2
        rows = torch.arange(z_dist.shape[0]).view(-1, 1).expand_as(nn_mask)
        # # # Extract values
        extracted_values = z_dist[rows, nn_mask]
        norm_values=extracted_values[:,-1].view(extracted_values.shape[0],1)
        # norm_values = torch.max(extracted_values, dim=1).values
        # norm_values = norm_values.view(extracted_values.shape[0],1)
        lid_Z = -torch.sum(torch.log10(extracted_values) - torch.log10(norm_values),axis=1)
        lid_nsa = sum(torch.square(lid_X - lid_Z))/(len(X)*self.k*10)
        return lid_nsa

class LID_NSALoss_v2(nn.Module):
    def __init__(self, k=5, eps=1e-7,full=False, **kwargs):
        super().__init__()
        self.k = k
        self.eps = eps
        self.full = full
    def compute_neighbor_mask(self, X, normA1):
        #print("Computing N neighbors for ground truth")
        x_dist = torch.cdist(X,X)+self.eps
        #print(x_dist)
        x_dist = x_dist/normA1
        # print(self.k,x_dist.shape)
        if self.full:
            values, indices = torch.topk(x_dist, x_dist.shape[0], largest=False)
        else:
            values, indices = torch.topk(x_dist, self.k+1, largest=False)
        values, indices = values[:,1:], indices[:,1:]
        #values = values/max(values)
        #print(values)
        norm_values=values[:,-1].view(values.shape[0],1)
        #print(norm_values)
        lid_X = torch.sum(torch.log10(values) - torch.log10(norm_values),axis=1)
        return indices, lid_X
    
    def forward(self, X, Z):
        normA1 = torch.quantile(torch.sqrt(torch.sum(X**2,axis=1)),0.98)
        normA2 = torch.quantile(torch.sqrt(torch.sum(Z**2,axis=1)),0.98)
        nn_mask, lid_X = self.compute_neighbor_mask(X, normA1)
        z_dist = torch.cdist(Z,Z)+self.eps
        z_dist = z_dist/normA2
        rows = torch.arange(z_dist.shape[0]).view(-1, 1).expand_as(nn_mask)
        # # # Extract values
        extracted_values = z_dist[rows, nn_mask]
        print(extracted_values.shape)
        # norm_values=extracted_values[:,-1].view(extracted_values.shape[0],1)
        norm_values = torch.max(extracted_values, dim=1).values
        norm_values = norm_values.view(extracted_values.shape[0],1)
        #print(norm_values)
        lid_Z = torch.sum(torch.log10(extracted_values) - torch.log10(norm_values),axis=1)
        lid_nsa = sum(torch.square(self.k/(lid_X+self.eps)-self.k/(lid_Z+self.eps)))/(len(X)*self.k*self.k)
        return lid_nsa


# class LID_NSALoss_v3(nn.Module):
#     def __init__(self, k=5, eps=1e-7, full=False, **kwargs):
#         super().__init__()
#         self.k = k
#         self.eps = eps
#         self.full = full
#     def compute_neighbor_mask(self, X, normA1):
#         #print("Computing N neighbors for ground truth")
#         x_dist = torch.cdist(X,X)+self.eps
#         #print(x_dist)
#         x_dist = x_dist/normA1
#         # print(self.k,x_dist.shape)
#         if self.full:
#             values, indices = torch.topk(x_dist, x_dist.shape[0], largest=False)
#         else:
#             values, indices = torch.topk(x_dist, self.k+1, largest=False)
#         values, indices = values[:,1:], indices[:,1:]
#         #values = values/max(values)
#         #print(values)
#         norm_values=values[:,-1].view(values.shape[0],1)
#         #print(norm_values)
#         lid_X = torch.log(torch.sum(torch.log10(values) - torch.log10(norm_values),axis=1))
#         print(lid_X)
#         return indices, lid_X
    
#     def forward(self, X, Z):
#         normA1 = torch.quantile(torch.sqrt(torch.sum(X**2,axis=1)),0.98)
#         normA2 = torch.quantile(torch.sqrt(torch.sum(Z**2,axis=1)),0.98)
#         nn_mask, lid_X = self.compute_neighbor_mask(X, normA1)
#         z_dist = torch.cdist(Z,Z)+self.eps
#         z_dist = z_dist/normA2
#         rows = torch.arange(z_dist.shape[0]).view(-1, 1).expand_as(nn_mask)
#         # # # Extract values
#         extracted_values = z_dist[rows, nn_mask]
#         norm_values=extracted_values[:,-1].view(extracted_values.shape[0],1)
#         # norm_values = torch.max(extracted_values, dim=1).values
#         # norm_values = norm_values.view(extracted_values.shape[0],1)
#         #print(extracted_values)
#         lid_Z = torch.log(torch.sum(torch.log10(extracted_values) - torch.log10(norm_values),axis=1))
#         print(lid_Z)
#         lid_nsa = sum(torch.square(lid_X - lid_Z))/(len(X)*self.k*10)
#         return lid_nsa

class LID_NSALoss_v3(nn.Module):
    def __init__(self, k=5, eps=1e-7, full=False, **kwargs):
        super().__init__()
        self.k = k
        self.eps = eps
        self.full = full
    def compute_neighbor_mask(self, X, normA1):
        #print("Computing N neighbors for ground truth")
        x_dist = torch.cdist(X,X)+self.eps
        #print(x_dist)
        x_dist = x_dist/normA1
        # print(self.k,x_dist.shape)
        if self.full:
            values, indices = torch.topk(x_dist, x_dist.shape[0], largest=False)
        else:
            values, indices = torch.topk(x_dist, self.k+1, largest=False)
        values, indices = values[:,1:], indices[:,1:]
        #values = values/max(values)
        #print(values)
        norm_values=values[:,-1].view(values.shape[0],1)
        #print(norm_values)
        lid_X = torch.sum(torch.log10(values) - torch.log10(norm_values),axis=1)
        return indices, lid_X


    def forward(self, X, Z):
        # normA1 = torch.quantile(torch.sqrt(torch.sum(X**2,axis=1)),0.98)
        # normA2 = torch.quantile(torch.sqrt(torch.sum(Z**2,axis=1)),0.98)

        mean_x = torch.mean(X, dim=0)
        mean_z = torch.mean(Z, dim=0)
        
        normA1 = torch.quantile(torch.sqrt(torch.sum((X - mean_x) ** 2, dim=1)),0.98)
        normA2 = torch.quantile(torch.sqrt(torch.sum((Z - mean_z) ** 2, dim=1)),0.98)

        nn_mask, lid_X = self.compute_neighbor_mask(X, normA1)
        z_dist = torch.cdist(Z,Z)+self.eps
        z_dist = z_dist/normA2
        rows = torch.arange(z_dist.shape[0]).view(-1, 1).expand_as(nn_mask)
        # # # Extract values
        extracted_values = z_dist[rows, nn_mask]
        norm_values=extracted_values[:,-1].view(extracted_values.shape[0],1)
        # norm_values = torch.max(extracted_values, dim=1).values
        # norm_values = norm_values.view(extracted_values.shape[0],1)
        #print(norm_values)
        lid_Z = torch.sum(torch.log10(extracted_values) - torch.log10(norm_values),axis=1)
        # print(lid_X[0:10])
        # print(lid_Z[0:10])
        # print(torch.square(torch.exp(lid_X/self.k) - torch.exp(lid_Z/self.k)))
        lid_nsa = sum(torch.square(torch.exp(lid_X/self.k) - torch.exp(lid_Z/self.k)))/(len(X)*10)
        return lid_nsa





class NSALoss3(nn.Module):
    def __init__(self, mode='raw', **kwargs):
        super().__init__()
        self.mode = mode
    
    def forward(self, x, z):
        # normA1 = torch.max(torch.sqrt(torch.sum(x**2,axis=1)))
        # normA2 = torch.max(torch.sqrt(torch.sum(z**2,axis=1)))
        if self.mode=='raw':
            mean_x = torch.mean(x, dim=0)
            mean_z = torch.mean(z, dim=0)
            
            normA1 = torch.quantile(torch.sqrt(torch.sum((x - mean_x) ** 2, dim=1)),0.98)
            normA2 = torch.quantile(torch.sqrt(torch.sum((z - mean_z) ** 2, dim=1)),0.98)
            # normA1 = torch.quantile(torch.sqrt(torch.sum(x**2,axis=1)),0.98)
            # normA2 = torch.quantile(torch.sqrt(torch.sum(z**2,axis=1)),0.98)
            
            A1_pairwise = torch.flatten(torch.cdist(x,x))    # compute pairwise dist
            A2_pairwise = torch.flatten(torch.cdist(z,z))    # compute pairwise dist
            A1_pairwise = A1_pairwise/(2*normA1)
            A2_pairwise = A2_pairwise/(2*normA2)
        elif self.mode=='dist':
            A1_pairwise = torch.flatten(x)
            A2_pairwise = torch.flatten(z)

        loss = torch.mean(torch.square(A2_pairwise - A1_pairwise))
        return loss