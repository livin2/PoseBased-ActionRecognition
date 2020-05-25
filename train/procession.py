import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from loguru import logger

@logger.catch
def train_epoch(model: nn.Module,
                iterator:DataLoader,
                eval_fn:callable,
                optimizer:optim.Optimizer,
                scheduler,
               batch_size):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for i_, batch in enumerate(iterator):
        optimizer.zero_grad()
        batch_loss,batch_acc = eval_fn(batch)
        epoch_loss+= batch_loss.item()
        batch_loss.backward()
        optimizer.step()
        epoch_loss+= batch_loss
        epoch_acc += batch_acc
    scheduler.step()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)/batch_size

@logger.catch
def evaluate(model: nn.Module,
             iterator:DataLoader,
             eval_fn:callable,
             batch_size):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for i_, batch in enumerate(iterator):
            batch_loss,batch_acc = eval_fn(batch)
            epoch_loss += batch_loss.item()
            epoch_acc += batch_acc
    return epoch_loss / len(iterator), epoch_acc / len(iterator)/batch_size

@logger.catch
def save_model(model,optimizer,loss,epoch,save_path='model'):
    print('SAVING EPOCH %d'%epoch)
    SAVE_FILE = os.path.join(save_path,'epoch_%d.pth'%epoch)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, SAVE_FILE)

@logger.catch
def save_model_with_info(model,optimizer,tagI2W,info,epoch,save_path='model'):
    print('SAVING EPOCH %d'%epoch)
    SAVE_FILE = os.path.join(save_path,'epoch_%d.pth'%epoch)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': info['train_loss'],
            'info': info,
            'tagI2W':tagI2W
            }, SAVE_FILE)

@logger.catch
def load_model_info(model,optimizer,checkpath):
    checkpoint = torch.load(checkpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    _epoch = checkpoint['epoch']
    _info = checkpoint['info']
    train_loss = _info['train_loss']
    train_acc = _info['train_acc']
    valid_loss = _info['valid_loss']
    valid_acc = _info['valid_acc']
    print('Epoch: %d' %(epoch))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')
    return checkpoint['tagI2W']

@logger.catch
def load_model(model,optimizer,checkpath):
    checkpoint = torch.load(checkpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    _epoch = checkpoint['epoch']
    _loss = checkpoint['loss']
    print('epoch:',_epoch)
    print('loss:',_loss)

@logger.catch
def load_model_eval(model,chept,map_location):
    checkpoint = torch.load(checkpath)
    model.load_state_dict(checkpoint['model_state_dict'],map_location=map_location)
    _epoch = checkpoint['epoch']
    _loss = checkpoint['loss']
    print('epoch:',_epoch)
    print('loss:',_loss)

import time

def print_train_info(start_time,epoch,train_loss,train_acc,valid_loss,valid_acc):
    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('\n',time.asctime( time.localtime(time.time())))
    print('Epoch: %d' %(epoch), " | time in %d minutes, %d seconds" %(mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')