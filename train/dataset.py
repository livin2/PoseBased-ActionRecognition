import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_sequence
class SeqDataset(Dataset):            
    def __init__(self, inputX,tagY):
        self.X = inputX
        self.Y = tagY
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return {'data':torch.Tensor(self.X[idx]),'tag':(self.Y[idx])}
    def collate_fn_pack(samples):
        datas = [s['data'] for s in samples]
        tags = [s['tag']  for s in samples]
        datas = pack_sequence(datas,enforce_sorted=False)
        return {'data':datas,'tag':tags}
    def collate_fn_pad(samples,batch_first=True):
        datas = [s['data'] for s in samples]
        lens =  [len(d) for d in datas]
        tags = [s['tag']  for s in samples]
        lens = torch.Tensor(lens)
        datas = pad_sequence(datas,batch_first=batch_first)
        return {'data':(datas,lens),'tag':tags}

class SinglePoseDataset(Dataset):            
    def __init__(self,inputX,tagY):
        self.X = inputX
        self.Y = tagY
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return {'data':torch.Tensor(self.X[idx]),'tag':(self.Y[idx])}
    # def __init__(self):
    #     self.X = []
    #     self.Y = []
    # def add(self,x,y):
    #     self.X.append(x)
    #     self.Y.append(y)
    # def numpy(self):
    #     self.X = np.array(self.X)
    #     self.Y = np.array(self.Y)