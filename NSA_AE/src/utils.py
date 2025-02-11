#!/usr/bin/python3
import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix, ConvexHull
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy.sparse.csgraph import connected_components, shortest_path
import sknetwork.path
from sklearn.exceptions import NotFittedError
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix
from itertools import chain

import gudhi as gd
import gudhi.hera as hera
import PIL

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import random
from scipy.sparse.csgraph import connected_components, shortest_path
#from src.utils import _fix_connected_components
import copy
import threading, queue
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import random
from scipy.sparse.csgraph import connected_components, shortest_path
#from src.utils import _fix_connected_components
import copy



def get_linear_model(input_dim, latent_dim=2, n_hidden_layers=2, hidden_dim=32, m_type='encoder', **kwargs):
    layers = list(
        chain.from_iterable(
            [
                (nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(n_hidden_layers)
            ]
        )
    )
    if m_type == 'encoder':
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()] + layers + [nn.Linear(hidden_dim, latent_dim)]
    elif m_type == 'decoder':
        layers = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()] + layers + [nn.Linear(hidden_dim, input_dim)]
    return nn.Sequential(*layers)

def get_cnn_model(input_dim=(64, 64), latent_dim=2, n_hidden_layers=2, hidden_dim=32, m_type='encoder', **kwargs):
    modules = []
    width, heigth = input_dim
    if m_type == 'encoder':
        in_channels = 1
        for i in range(n_hidden_layers):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=hidden_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.LeakyReLU())
            )
            in_channels = hidden_dim
        modules.append(nn.Flatten(start_dim=1, end_dim=- 1))
        modules.append(nn.Linear(int(hidden_dim*width*heigth/(4**n_hidden_layers)), latent_dim))
    elif m_type == 'decoder':
        shape = int(hidden_dim*width*heigth/(4**n_hidden_layers))
        modules.append(nn.Linear(latent_dim, shape))
        modules.append(Reshape(hidden_dim, int(width/(2**n_hidden_layers)), int(heigth/(2**n_hidden_layers))))
        for i in range(n_hidden_layers-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dim, hidden_dim,
                              kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.LeakyReLU())
            )
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dim, 1,
                          kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(1),
                nn.LeakyReLU())
        )
    return nn.Sequential(*modules)
            

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view((batch_size, *self.shape))

def get_geodesic_distance(data, n_neighbors=3, **kwargs):
    kng = kneighbors_graph(data, n_neighbors=n_neighbors, mode='distance', **kwargs)
    n_connected_components, labels = connected_components(kng)
    if n_connected_components > 1:
        kng = _fix_connected_components(
                    X=data,
                    graph=kng,
                    n_connected_components=n_connected_components,
                    component_labels=labels,
                    mode="distance",
                    **kwargs
                )

    if connected_components(kng)[0] != 1:
        raise ValueError("More than 1 connected component in the end!")
    #     return shortest_path(kng, directed=False)
    print(f"N connected: {n_connected_components}")
    return shortest_path(kng, directed=False)




class FromNumpyDataset(Dataset):
    def __init__(self, data, labels=None, geodesic=False, flatten=True, scaler=None, **kwargs):
        if labels is not None:
            assert len(labels) == len(data), "The length of labels and data are not equal"
            self.labels = labels
        if flatten:
            self.data = torch.tensor(data).flatten(start_dim=1).numpy()
        else:
            self.data = data
        if scaler is not None:
            try:
                self.data = scaler.transform(self.data)
            except NotFittedError:
                self.data = scaler.fit_transform(self.data)
        self.scaler = scaler
        if geodesic:
            self.data_dist = get_geodesic_distance(self.data, **kwargs)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if hasattr(self, 'labels'):
            label = self.labels[idx]
        else:
            label = 0
        if hasattr(self, 'data_dist'):
            return idx, self.data[idx], label, self.data_dist[idx]
        else:
            return idx, self.data[idx], label
        
def get_latent_representations(model, data_loader):
    labels = []
    data = []
    model.eval()
    model.to('cpu')
    with torch.no_grad():
        for x, _, y in data_loader:
#             if x.device != model.device:
#                 x.to(model.device)
            labels.append(y.numpy())
            data.append(model(x).cpu().numpy())
    return np.concatenate(data, axis=0), np.concatenate(labels, axis=0)

def get_output_representations(model, data_loader):
    labels = []
    data = []
    model.eval()
    model.to('cpu')
    with torch.no_grad():
        for x, _, y in data_loader:
#             if x.device != model.device:
#                 x.to(model.device)
            labels.append(y.numpy())
            z=model.encoder(x)
            data.append(model.decoder(z).cpu().numpy())
    return np.concatenate(data, axis=0), np.concatenate(labels, axis=0)

def vizualize_data(data, labels=None, alpha=1.0, s=1.0, title="", ax=None):
    assert labels.shape[0] == data.shape[0], "Length of labels and data are not equal"
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))
    if data.shape[1] == 2:
        x, y = zip(*data)
        ax.scatter(x, y, alpha=alpha, c=labels, s=s)
    else:
        x, y, z = zip(*data)
        ax.scatter(x, y, z, alpha=alpha, c=labels, s=s)
    ax.set_title(title, fontsize=20)
    return ax

def plot_latent_tensorboard(latent, labels):
    if latent.shape[1] < 3:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(latent[:, 0], latent[:, 1], c=labels, s=20.0, alpha=0.7, cmap='viridis')
    elif latent.shape[1] == 3:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(latent[:, 0], latent[:, 1], latent[:, 2], c=labels, s=1.0, alpha=0.7, cmap='viridis')
    else:
        return None
    fig.canvas.draw()
    image = np.array(PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb()))
    plt.close(fig)
    return image
#     return np.swapaxes(np.array(fig.canvas.renderer.buffer_rgba()), -1, 1)

def plot_latent(train_latent, train_labels, model_name, dataset_name):
    if train_latent.shape[1] > 2:
        fig = plt.figure(figsize=(12, 8))
        axes = fig.add_subplot(1, 1, 1, projection='3d')
    else:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    axes = vizualize_data(train_latent, train_labels, title=f"Model: {model_name}, dataset:{dataset_name}", ax=axes)
    return fig, axes

def calculate_barcodes(distances, max_dim=1):
    skeleton = gd.RipsComplex(distance_matrix = distances)
    simplex_tree = skeleton.create_simplex_tree(max_dimension=max_dim+1)
    barcodes = simplex_tree.persistence()
    pbarcodes = {}
    for i in range(max_dim+1):
        pbarcodes[i] = [[b[1][0], b[1][1]] for b in barcodes if b[0] == i]
    return pbarcodes

def cast_to_normal_array(barcodes):
    return np.array([[b, d] for b, d in barcodes])

def calculate_wasserstein_distance(x, z, n_runs = 5, batch_size = 2048, max_dim = 1):
    if batch_size > len(x):
        n_runs = 1
    
    results = {d:[] for d in range(max_dim+1)}
    x = x.reshape(len(x), -1)
    z = z.reshape(len(z), -1)
    for i in range(n_runs):
        ids = np.random.choice(np.arange(0, len(x)), size=min(batch_size, len(x)), replace=False)
        data = x[ids]
        distances = distance_matrix(data, data)
        distances = distances/np.percentile(distances.flatten(), 90)
        
        barcodes = {'original':calculate_barcodes(distances, max_dim=max_dim)}
        
        data = z[ids]
        distances = distance_matrix(data, data)
        distances = distances/np.percentile(distances.flatten(), 90)
        barcodes['model'] = calculate_barcodes(distances, max_dim=max_dim)
        for dim in range(max_dim+1):
            original = cast_to_normal_array(barcodes['original'][dim])
            model = cast_to_normal_array(barcodes['model'][dim])
            results[dim].append(hera.wasserstein_distance(original, model, internal_p=1))
    return results
         



class FurthestScaler:
    def __init__(self, p=2): # approximate
        self.is_fitted = False
        self.p = p
        
    def fit(self, data):
        self.furthest = self._furthest_distance(data)
        self.is_fitted = True
        
    def transform(self, data):
        if not self.is_fitted:
            raise NotFittedError
        return data / self.furthest
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def _furthest_distance(self, points, sample_frac=0.0):
        # exact solution, very computationaly expesive
        # hull = ConvexHull(points) 
        # hullpoints = points[hull.vertices,:]
        # hdist = distance_matrix(hullpoints, hullpoints, p=self.p)
        # approximation: upper bound
        # pick random point and compute distances to all of the points
        # diameter min: max(distances), diameter max (triangle inequality): 2 max(distances)
        if len(points.shape) > 2:
            points = points.reshape(points.shape[0],-1)
        idx = np.random.choice(np.arange(len(points)), size=1)
        hdist = distance_matrix(points[idx], points, p=self.p)
        return 0.1*hdist.max() # upper bound


class priorityQ_torch(object):
    """Priority Q implelmentation in PyTorch

    Args:
        object ([torch.Tensor]): [The Queue to work on]
    """

    def __init__(self, val):
        self.q = torch.tensor([[val, 0]])
        # self.top = self.q[0]
        # self.isEmpty = self.q.shape[0] == 0

    def push(self, x):
        """Pushes x to q based on weightvalue in x. Maintains ascending order

        Args:
            q ([torch.Tensor]): [The tensor queue arranged in ascending order of weight value]
            x ([torch.Tensor]): [[index, weight] tensor to be inserted]

        Returns:
            [torch.Tensor]: [The queue tensor after correct insertion]
        """
        if type(x) == np.ndarray:
            x = torch.tensor(x)
        if self.isEmpty():
            self.q = x
            self.q = torch.unsqueeze(self.q, dim=0)
            return
        idx = torch.searchsorted(self.q.T[1], x[1])
        #print(idx)
        self.q = torch.vstack([self.q[0:idx], x, self.q[idx:]]).contiguous()

    def top(self):
        """Returns the top element from the queue

        Returns:
            [torch.Tensor]: [top element]
        """
        return self.q[0]

    def pop(self):
        """pops(without return) the highest priority element with the minimum weight

        Args:
            q ([torch.Tensor]): [The tensor queue arranged in ascending order of weight value]

        Returns:
            [torch.Tensor]: [highest priority element]
        """
        if self.isEmpty():
            print("Can Not Pop")
        self.q = self.q[1:]

    def isEmpty(self):
        """Checks is the priority queue is empty

        Args:
            q ([torch.Tensor]): [The tensor queue arranged in ascending order of weight value]

        Returns:
            [Bool] : [Returns True is empty]
        """
        return self.q.shape[0] == 0

def dijkstra(adj):
    n = adj.shape[0]
    distance_matrix = torch.zeros([n, n])
    for i in range(n):
        u = torch.zeros(n, dtype=torch.bool)
        d = np.inf * torch.ones(n)
        d[i] = 0
        q = priorityQ_torch(i)
        while not q.isEmpty():
            v, d_v = q.top()  # point and distance
            v = v.int()
            q.pop()
            if d_v != d[v]:
                continue
            for j, py in enumerate(adj[v]):
                if py == 0 and j != v:
                    continue
                else:
                    to = j
                    weight = py
                    if d[v] + py < d[to]:
                        d[to] = d[v] + py
                        q.push(torch.Tensor([to, d[to]]))
        distance_matrix[i] = d
    return distance_matrix





def get_geodesic_distance_torch(data, n_neighbors=3, **kwargs):
    # Convert data to a numpy array if it's a torch tensor
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    kng = kneighbors_graph(data, n_neighbors=n_neighbors, mode='distance', **kwargs)
    n_connected_components, labels = connected_components(kng)
    if n_connected_components > 1:
        kng = _fix_connected_components_torch(
                    X=torch.tensor(data),
                    graph=kng,
                    n_connected_components=n_connected_components,
                    component_labels=labels,
                    mode="distance",
                    **kwargs
                )

    if connected_components(kng)[0] != 1:
        raise ValueError("More than 1 connected component in the end!")
    print(f"N connected: {n_connected_components}")
    return dijkstra(torch.tensor(kng.todense()))


class NearestNeighborBatchSamplerMulti(Sampler):
    def __init__(self, dataset, batch_size, num_neighbors, num_threads=24):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors
        #Given input data X, compute k nearest neighbors for each point
        self.kng = kneighbors_graph(self.dataset.data, n_neighbors=self.num_neighbors, mode='connectivity')
        self.kng_dict_front = self.create_kng_dict_front()
        #Use fix connected components to generate one connected component
        num_ccs, component_labels = connected_components(self.kng)
        if num_ccs > 1:
            self.kng = _fix_connected_components(self.dataset.data, self.kng, num_ccs, component_labels, mode='connectivity', metric='euclidean')
            #Check again to confirm if it worked
            num_ccs, component_labels = connected_components(self.kng)
            if num_ccs >1:
                raise ValueError("Increase nearest neighbor size; cannot generate a single connected component with the given knn size.")
        self.results_queue = queue.Queue()
        self.num_threads = num_threads
        self.indices = []

    def create_kng_dict_front(self):
        dict = [list(self.nearest_neighbors(i)) for i in range(len(self.dataset))]
        return dict
        
    def create_kng_dict_back(self):
        dict = [list(self.nearest_neighbors_of(i)) for i in range(len(self.dataset))]
        return dict

    def nearest_neighbors(self, index, kng=None):
        if kng is None:
            kng = self.kng
        return kng.getrow(index).nonzero()[1]

    def nearest_neighbors_of(self, index, kng=None):
        if kng is None:
            kng = self.kng
        return kng.getcol(index).nonzero()[0]

    def one_minibatch(self, point):
        batch = set()
        point_queue = []
        queue_set = set()
        batch.add(point)
        point_queue.extend(self.kng_dict_front[point])
        queue_set.update(self.kng_dict_front[point])
        while len(batch) + len(point_queue) < self.batch_size:
            if point_queue:
                next_point = point_queue.pop(0)
                if next_point not in batch:
                    batch.add(next_point)
                    for new_point in self.kng_dict_front[next_point]:
                        if new_point not in queue_set:
                            point_queue.append(new_point)
                            queue_set.add(new_point)
            else:
                if self.indices:
                    next_point = self.indices.pop()
                    batch.add(next_point)
                    point_queue.extend(self.kng_dict_front[next_point])
                    queue_set.update(self.kng_dict_front[next_point])
        batch.update(point_queue)
        return batch
        
    def thread_minibatch(self):
        while True:
            try:
                point = self.indices.pop()  # Get a value to process
            except IndexError:
                break
            batch = self.one_minibatch(point)
            self.results_queue.put(batch)
    
    def __iter__(self):
        #print("Running iter")
        numbatches = len(self.dataset) // self.batch_size
        self.indices = list(range(len(self.dataset)))
        random.shuffle(self.indices)
        self.indices = self.indices[:numbatches]
        batches = []
        threads = []
        for _ in range(self.num_threads):
            thread = threading.Thread(target=self.thread_minibatch)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        while not self.results_queue.empty():
            result = self.results_queue.get()
            batches.append(list(result))
        return iter(batches)

    def __len__(self):
        return len(self.dataset) // self.batch_size

class NearestNeighborBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_neighbors):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors
        #Given input data X, compute k nearest neighbors for each point
        self.kng = kneighbors_graph(self.dataset.data, n_neighbors=self.num_neighbors, mode='connectivity')
        self.kng_dict_front = self.create_kng_dict_front()
        self.kng_dict_back = self.create_kng_dict_back()
        #Use fix connected components to generate one connected component
        num_ccs, component_labels = connected_components(self.kng)
        if num_ccs > 1:
            self.kng = _fix_connected_components(self.dataset.data, self.kng, num_ccs, component_labels, mode='connectivity', metric='euclidean')
            #Check again to confirm if it worked
            num_ccs, component_labels = connected_components(self.kng)
            if num_ccs >1:
                raise ValueError("Increase nearest neighbor size; cannot generate a single connected component with the given knn size.")

    def create_kng_dict_front(self):
        dict = [list(self.nearest_neighbors(i)) for i in range(len(self.dataset))]
        return dict
        
    def create_kng_dict_back(self):
        dict = [list(self.nearest_neighbors_of(i)) for i in range(len(self.dataset))]
        return dict

    def nearest_neighbors(self, index, kng=None):
        if kng is None:
            kng = self.kng
        return kng.getrow(index).nonzero()[1]

    def nearest_neighbors_of(self, index, kng=None):
        if kng is None:
            kng = self.kng
        return kng.getcol(index).nonzero()[0]
        
    # def remove_point_existence(self, index):
    #     for i in self.kng_dict_back[index]:
    #         self.kng_dict_front[i].remove(index)
    #     for i in self.kng_dict_front[index]:
    #         self.kng_dict_back[i].remove(index)
    #     self.kng_dict_front[index] = []
    #     self.kng_dict_back[index] = []
    
    def __iter__(self):
        #print("Running iter")
        # iter_kng = self.kng.copy()
        numbatches = len(self.dataset) // self.batch_size
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        indices = indices[:numbatches]
        # self.kng_dict_front = self.create_kng_dict_front()
        # self.kng_dict_back = self.create_kng_dict_back()
        # indices = set(indices)
        batches = []
        while indices:
            kng_front = copy.deepcopy(self.kng_dict_front)
            kng_back = copy.deepcopy(self.kng_dict_back)
            # if len(indices) < self.batch_size:
            #     #Add functionality to sort the data in case you want to retain ordering
            #     batches.append(indices)
            #     break
            # print(indices[:10])
            batch = set()
            queue = []
            queue_set = set()
            next_point = indices.pop()
            # print("Current index:",next_point)
            # print("Remaining indices:",len(indices))
            # print("Number of batches:", len(batches))
            batch.add(next_point)
            queue.extend(self.kng_dict_front[next_point])
            queue_set.update(self.kng_dict_front[next_point])
            for i in kng_back[next_point]:
                kng_front[i].remove(next_point)
            for i in kng_front[next_point]:
                kng_back[i].remove(next_point)
            kng_front[next_point] = []
            kng_back[next_point] = []
            # self.remove_point_existence(next_point)
            while len(batch) + len(queue) < self.batch_size:
                # print("Queue Length:",len(queue))
                if queue:
                    next_point = queue.pop(0)
                    if next_point not in batch:
                        batch.add(next_point)
                        # indices.remove(next_point)
                        for new_point in self.kng_dict_front[next_point]:
                            if new_point not in queue_set:
                                queue.append(new_point)
                                queue_set.add(new_point)
                        for i in kng_back[next_point]:
                            kng_front[i].remove(next_point)
                        for i in kng_front[next_point]:
                            kng_back[i].remove(next_point)
                        kng_front[next_point] = []
                        kng_back[next_point] = []
                        # queue.extend(self.kng_dict_front[next_point])
                        # self.remove_point_existence(next_point)
                else:
                #     # print("Queue is empty but batch requirements are not met starting from point:",next_point)
                    if indices:
                        next_point = indices.pop()
                        batch.add(next_point)
                        queue.extend(self.kng_dict_front[next_point])
                        queue_set.update(self.kng_dict_front[next_point])
                        for i in kng_back[next_point]:
                            kng_front[i].remove(next_point)
                        for i in kng_front[next_point]:
                            kng_back[i].remove(next_point)
                        kng_front[next_point] = []
                        kng_back[next_point] = []
                        # queue.extend(self.kng_dict_front[next_point])
                        # self.remove_point_existence(next_point)
                # Remove duplicates in queue
                #queue = list(dict.fromkeys(queue))
            batch.update(queue)
            # print("Created a batch of size:",len(batch))
        
            batches.append(list(batch))
            # print("Previous batch size:", len(batches[-1]))
            #This might not be necessary
            # num_ccs, component_labels = connected_components(iter_kng)
            # if num_ccs > 1:
            #     print("Connected component check failed during batching")
            #     # iter_kng = _fix_connected_components(self.dataset.data, iter_kng, num_ccs, component_labels, mode='connectivity', metric='euclidean')
            #     # #Check to see if it was fixed
            #     # num_ccs, component_labels = connected_components(iter_kng)
            #     # if n_ccs >1:
            #     #     raise ValueError("Connected component check failed during batching; increase nearest neighbor size.")
        
        return iter(batches)

    def __len__(self):
        return len(self.dataset) // self.batch_size

class CustomDataLoader(DataLoader):
    def __init__(self, dataset, k, **kwargs):
        super().__init__(dataset, **kwargs)
        self.k = k
        self.epoch = 0
    def __iter__(self):
        if self.epoch % self.k == 0:
            self._iterator = self._get_iterator()
            self.epoch = 0  # Reset the epoch counter
        self.epoch += 1
        return self._iterator


def _fix_connected_components_torch(
    X,
    graph,
    n_connected_components,
    component_labels,
    mode="distance",
    metric="euclidean",
    **kwargs,
):
    # X is expected to be a torch tensor
    if metric == "precomputed":
        raise NotImplementedError("_fix_connected_components_torch does not support 'precomputed' metric.")

    for i in range(n_connected_components):
        idx_i = (component_labels == i).nonzero().squeeze(1)
        Xi = X[idx_i]
        for j in range(i):
            idx_j = (component_labels == j).nonzero().squeeze(1)
            Xj = X[idx_j]

            # Compute pairwise distances using PyTorch
            D = torch.cdist(Xi, Xj, p=2)  # Euclidean distance

            ii, jj = torch.unravel_index(D.argmin(), D.shape)
            if mode == "connectivity":
                graph[idx_i[ii], idx_j[jj]] = 1
                graph[idx_j[jj], idx_i[ii]] = 1
            elif mode == "distance":
                graph[idx_i[ii], idx_j[jj]] = D[ii, jj].item()
                graph[idx_j[jj], idx_i[ii]] = D[ii, jj].item()
            else:
                raise ValueError("Unknown mode={}, should be one of ['connectivity', 'distance'].".format(mode))

    return graph


def _fix_connected_components(
    X,
    graph,
    n_connected_components,
    component_labels,
    mode="distance",
    metric="euclidean",
    **kwargs,
):
    """Add connections to sparse graph to connect unconnected components.
    For each pair of unconnected components, compute all pairwise distances
    from one component to the other, and add a connection on the closest pair
    of samples. This is a hacky way to get a graph with a single connected
    component, which is necessary for example to compute a shortest path
    between all pairs of samples in the graph.
    Parameters
    ----------
    X : array of shape (n_samples, n_features) or (n_samples, n_samples)
        Features to compute the pairwise distances. If `metric =
        "precomputed"`, X is the matrix of pairwise distances.
    graph : sparse matrix of shape (n_samples, n_samples)
        Graph of connection between samples.
    n_connected_components : int
        Number of connected components, as computed by
        `scipy.sparse.csgraph.connected_components`.
    component_labels : array of shape (n_samples)
        Labels of connected components, as computed by
        `scipy.sparse.csgraph.connected_components`.
    mode : {'connectivity', 'distance'}, default='distance'
        Type of graph matrix: 'connectivity' corresponds to the connectivity
        matrix with ones and zeros, and 'distance' corresponds to the distances
        between neighbors according to the given metric.
    metric : str
        Metric used in `sklearn.metrics.pairwise.pairwise_distances`.
    kwargs : kwargs
        Keyword arguments passed to
        `sklearn.metrics.pairwise.pairwise_distances`.
    Returns
    -------
    graph : sparse matrix of shape (n_samples, n_samples)
        Graph of connection between samples, with a single connected component.
    """
    if metric == "precomputed" and sparse.issparse(X):
        raise RuntimeError(
            "_fix_connected_components with metric='precomputed' requires the "
            "full distance matrix in X, and does not work with a sparse "
            "neighbors graph."
        )

    for i in range(n_connected_components):
        idx_i = np.flatnonzero(component_labels == i)
        Xi = X[idx_i]
        for j in range(i):
            idx_j = np.flatnonzero(component_labels == j)
            Xj = X[idx_j]

            if metric == "precomputed":
                D = X[np.ix_(idx_i, idx_j)]
            else:
                D = pairwise_distances(Xi, Xj, metric=metric, **kwargs)

            ii, jj = np.unravel_index(D.argmin(axis=None), D.shape)
            if mode == "connectivity":
                graph[idx_i[ii], idx_j[jj]] = 1
                graph[idx_j[jj], idx_i[ii]] = 1
            elif mode == "distance":
                graph[idx_i[ii], idx_j[jj]] = D[ii, jj]
                graph[idx_j[jj], idx_i[ii]] = D[ii, jj]
            else:
                raise ValueError(
                    "Unknown mode=%r, should be one of ['connectivity', 'distance']."
                    % mode
                )

    return graph