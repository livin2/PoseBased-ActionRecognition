import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence 
from torch.nn.utils.rnn import pad_packed_sequence

def pack_lstm_pad(lstm,_in,_len,_hidden=None,batch_first=True):
    pack_in = pack_padded_sequence(_in, _len, batch_first,enforce_sorted=False)
    pack_out,hc = lstm(pack_in,_hidden)
    pad_out,_len = pad_packed_sequence(pack_out,batch_first)
    return pad_out,hc

def debuglog(obj):
    print(obj.shape)
    if(len(obj.shape)==3):
        print(obj[0,:2,:3])
    if(len(obj.shape)==2):
        print(obj[:2,:3])
    print('')

POSE_JOINT_SIZE = 34
def get_model(name,tagI2W):
    if(name == 'FcLstm'):
        return FcLstm(POSE_JOINT_SIZE,len(tagI2W))
    elif(name == 'dnnSingle'):
        return dnnSingle(POSE_JOINT_SIZE,len(tagI2W))

class FcLstm(nn.Module):
    def __init__(self,input_dim,class_num,batch_first=True,initrange=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(128, 64)
        self.lstm = nn.LSTM(64,64,batch_first=batch_first)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, class_num)
        self.init_weights(initrange)
        self.batch_first = batch_first

    def init_weights(self,initrange):
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc3.weight.data.uniform_(-initrange, initrange)
        self.fc3.bias.data.zero_()
        self.fc4.weight.data.uniform_(-initrange, initrange)
        self.fc4.bias.data.zero_()

    def forward(self, _input,_len,_hidden):
        _fc1   = self.fc1(_input)
        _fc2   = self.fc2(_fc1)
        _lstm,hc = pack_lstm_pad(self.lstm,_fc2,_len,_hidden,self.batch_first)
        _lstm_m = F.adaptive_max_pool2d(_lstm,[1,64])
        _fc3   = self.fc3(_lstm_m)
        _fc4   = self.fc4(_fc3)
        _out = F.softmax(_fc4,dim=2)
        # return F.softmax(_fc4),hc
        return _out.view(-1,_out.shape[-1]),hc

    def exe_pre(self,device,holder):
        holder.hc_ = None
        holder.len_ = torch.Tensor([1]).to(device)
    
    def exe(self,input_,device,holder):
        input_ = torch.Tensor(input_.reshape(1,1,-1)).to(device)
        if(holder.hc_ is None):
            holder.hc_ = (torch.randn(1,1,64).to(device), torch.randn(1,1,64).to(device))#LSTM out
        out,_hc = self.__call__(input_,holder.len_,holder.hc_) # data,datalen
        holder.hc_ = (_hc[0].data,_hc[1].data)
        return out

class dnnSingle(nn.Module):
    def __init__(self,input_dim,class_num,batch_first=True,initrange=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, class_num)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(16)
        self.class_num = class_num
        
        self.init_weights(initrange)
        self.batch_first = batch_first

    def init_weights(self,initrange):
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc3.weight.data.uniform_(-initrange, initrange)
        self.fc3.bias.data.zero_()
        self.fc4.weight.data.uniform_(-initrange, initrange)
        self.fc4.bias.data.zero_()

    def forward(self, _input):
        _fc1   = F.relu(self.fc1(_input))
        _bn1 = self.bn1(_fc1)
        _fc2   = F.relu(self.fc2(_bn1))
        _bn2 = self.bn2(_fc2)
        _fc3   = F.relu(self.fc3(_bn2))
        _bn3 = self.bn3(_fc3)
        _fc4   = F.relu(self.fc4(_bn3))

        # debuglog(_fc1)
        # debuglog(_bn1)
        # debuglog(_fc3)
        # debuglog(_fc4)
        _out = F.softmax(_fc4,dim=1) #single
        # _out = F.softmax(_fc4,dim=2) #seq
        # debuglog(_out)
        # print('sm2:',_out.sum(2))
        # print('')
        # print('sm:_out = F.softmax(_fc4,dim=1)out)

        return _out.view(-1,_out.shape[-1])

    def exe_pre(self,device,holder):
        pass

    def exe(self,input_,device,holder):
        input_ = torch.Tensor(input_).to(device)
        return self.__call__(input_) # data,datalen


# class StLstm(nn.Module): pass

class dnnSeq(nn.Module):
    def __init__(self,input_dim,class_num,batch_first=True,initrange=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        # self.lstm = nn.LSTM(64,64,batch_first=batch_first)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, class_num)
        self.class_num = class_num
        
        self.init_weights(initrange)
        self.batch_first = batch_first

    def init_weights(self,initrange):
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc3.weight.data.uniform_(-initrange, initrange)
        self.fc3.bias.data.zero_()
        self.fc4.weight.data.uniform_(-initrange, initrange)
        self.fc4.bias.data.zero_()

    def forward(self, _input):
        _fc1   = F.relu(self.fc1(_input))
        lnorm1 = nn.LayerNorm(_fc1.size()[1:]).cuda()
        _fc2   = F.relu(self.fc2(lnorm1(_fc1)))
        lnorm2 = nn.LayerNorm(_fc2.size()[1:]).cuda()
        _fc3   = F.relu(self.fc3(lnorm2(_fc2)))
        lnorm3 = nn.LayerNorm(_fc3.size()[1:]).cuda()
        _fc4   = F.relu(self.fc4(lnorm3(_fc3)))

        debuglog(_fc1)
        debuglog(_fc2)
        debuglog(_fc3)
        debuglog(_fc4)

        # _out = F.adaptive_max_pool2d(_out,output_size=(1,self.class_num))
        _out = F.softmax(_fc4,dim=2) #single
        # _out = F.softmax(_fc4,dim=2) #seq
        # debuglog(_out)
        # print('sm2:',_out.sum(2))
        # print('')
        # print('sm:_out = F.softmax(_fc4,dim=1)out)

        return _out.view(-1,_out.shape[-1])