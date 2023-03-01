import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.debug.metrics as met

import torchvision
import torchvision.transforms as transforms

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.mp1 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1440, 10)
    
    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = self.mp1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.Softmax(dim=-1)(x)
        return x


FLAGS = {'batch_size':64,
         'world_size':8,
         'epochs':10,
         'log_steps':10,
         'metrics_debug':False}


def train(rank, FLAGS):
    print(f'Starting train method on rank: {rank}')
    #dist.init_process_group(backend='nccl', world_size=FLAGS['world_size'], init_method='env://', rank=rank)
    
    device = xm.xla_device()
    WRAPPED_MODEL = xmp.MpModelWrapper(ToyModel())
    model = WRAPPED_MODEL.to(device)
    #torch.cuda.set_device(rank)
    #model.cuda(rank)
    
    criterion = nn.CrossEntropyLoss().cuda(rank)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    
    #model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    transform = transforms.Compose([torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307,),(0.3081,))
                                    ])
    train_dataset = torchvision.datasets.MNIST('/tmp', train=True, download=True, transform=transform)
    print(len(train_dataset))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=FLAGS['world_size'], rank=rank)
    
    for epoch in range(FLAGS['epochs']):
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS['batch_size'], shuffle=False, num_workers=0, sampler=train_sampler)
        
        para_loader = pl.ParallelLoader(train_loader, [device])
        device_loader = para_loader.per_device_loader(device)
        
        for i, (images, labels) in enumerate(device_loader):
            #images = images.cuda(non_blocking=True)
            #labels = labels.cuda(non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            
            #optimizer.step()
            xm.optimizer_step(optimizer)
            
            if not i % FLAGS['log_steps']:
                total_epochs = FLAGS['epochs']
                #print(f'Epoch: {epoch+1}/{total_epochs}, Loss:{loss.item()}')
                xm.master_print(f'Epoch: {epoch+1}/{total_epochs}, Loss:{loss.item()}')
                
        if FLAGS['metrics_debug']:
            xm.master_print(met.metrics_report())
        else:
            xm.master_print('------------')


if __name__ == '__main__':    
    xmp.spawn(train, nprocs=FLAGS['world_size'], args=(FLAGS,), start_method='fork')

