from dgl.data import DGLDataset
import torch
import dgl
import os
from utils import load_data

os.environ["DGLBACKEND"] = "pytorch"

class TwitterNewsDataLoader(DGLDataset):
    def __init__(self):
        super().__init__(name="twitter_news")

    def create_mask(self, idxs, N):
        result = [0] * N
        for i in idxs:
            result[i] = 1
        return torch.LongTensor(result)
    
    def process(self):
        args = {
            'dataset': 'newsbias_event_1_untyped'
        }
        raw_data, data = load_data(args, '.', False, "unsup1")

        N = data['num_nodes']
        srcs = []
        dests = []
        for src in data['node2adj']:
            for dest in data['node2adj'][src]:
                srcs.append(src)
                dests.append(dest)

        self.graph = dgl.graph(
            (srcs, dests), num_nodes=N
        )
        self.graph.ndata["label"] = torch.LongTensor(data[''])
        self.graph.ndata["train_mask"] = self.create_mask(data['idx_train'], N)
        self.graph.ndata["val_mask"] = self.create_mask(data['idx_valid'], N)
        self.graph.ndata["test_mask"] = self.create_mask(data['idx_test'], N)

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1
