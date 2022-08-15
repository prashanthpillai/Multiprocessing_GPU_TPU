import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

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
        

FLAGS = {'batch_size':200,
         'world_size':4,
         'epochs':50,
         'log_steps':100}
         
def train(rank, FLAGS):
    print(f'Starting train method on rank: {rank}')
    dist.init_process_group(backend='nccl', world_size=FLAGS['world_size'], init_method='env://', rank=rank)
    
    model = ToyModel()
    torch.cuda.set_device(rank)
    model.cuda(rank)
    
    criterion = nn.CrossEntropyLoss().cuda(rank)
    optimizer = torch.optim.SGD(model.parameters(), 1e-3)
    
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    transform = transforms.Compose([torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307,),(0.3081,))
                                    ])
    train_dataset = torchvision.datasets.MNIST('/tmp', train=True, download=True, transform=transform)
    print(len(train_dataset))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=FLAGS['world_size'], rank=rank)
    
    for epoch in range(FLAGS['epochs']):
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS['batch_size'], shuffle=False, num_workers=0, sampler=train_sampler)
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not i % FLAGS['log_steps']:
                total_epochs = FLAGS['epochs']
                print(f'Epoch: {epoch+1}/{total_epochs}, Loss:{loss.item()}')
            

if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # set to DETAIL for runtime logging.
    mp.spawn(train, nprocs=FLAGS['world_size'], args=(FLAGS,))