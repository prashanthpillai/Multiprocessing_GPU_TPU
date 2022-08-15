import os
import torch
import torch.nn as nn
import transformers
import numpy as np
import pandas as pd
from sklearn import model_selection
from transformers import AdamW, get_linear_schedule_with_warmup
from scipy import stats

import torch.multiprocessing as mp
import torch.distributed as dist

import warnings
warnings.filterwarnings(action='ignore')
transformers.logging.set_verbosity_error()

class BERTBaseUncased(nn.Module):
    
    def __init__(self, bert_path):
        super(BERTBaseUncased, self).__init__()
        self.bert_path = bert_path
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 30)
        
    def forward(self, ids, mask, token_type_ids):
        bert_out = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)    
        pooler_out = bert_out[1]
        bo = self.bert_drop(pooler_out)
        return self.out(bo)


class BERTDatasetTraining:
    def __init__(self, qtitle, qbody, answer, targets, tokenizer, max_len):
        self.qtitle = qtitle
        self.qbody = qbody
        self.answer = answer
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.targets = targets
        
    def __len__(self):
        return len(self.answer)
    
    def __getitem__(self, item):
        question_title = str(self.qtitle[item])
        question_body = str(self.qbody[item])
        answer = str(self.answer[item])
        
        inputs = self.tokenizer.encode_plus(
        question_title + " " + question_body,
        answer,
        add_special_tokens=True,	
        max_length=self.max_len, truncation=True)
        
        ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']
        mask = inputs['attention_mask']
        
        padding_len = self.max_len - len(ids)
        ids = ids + ([0]) * padding_len
        token_type_ids = token_type_ids + ([0]) * padding_len
        mask = mask + ([0]) * padding_len
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets':torch.tensor(self.targets[item, :], dtype=torch.float)
        }

def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)

def train_loop_fn(data_loader, model, optimizer, scheduler=None):
    model.train()
    for bi, d in enumerate(data_loader):
        ids = d['ids']
        mask = d['mask']
        token_type_ids = d['token_type_ids']
        targets = d['targets']
        
        #ids = ids.to(device, dtype=torch.long)
        #mask = mask.to(device, dtype=torch.long)
        #token_type_ids = token_type_ids.to(device, dtype=torch.long)
        #targets = targets.to(device, dtype=torch.float)
        
        ids = ids.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        token_type_ids = token_type_ids.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if bi % 10 == 0:
            print(f'bi={bi}, loss={loss}')

def eval_loop_fn(data_loader, model):
    model.eval()
    fin_targets = []
    fin_outputs = []
    for bi, d in enumerate(data_loader):
        ids = d['ids']
        mask = d['mask']
        token_type_ids = d['token_type_ids']
        targets = d['targets']
        
        #ids = ids.to(device, dtype=torch.long)
        #mask = mask.to(device, dtype=torch.long)
        #token_type_ids = token_type_ids.to(device, dtype=torch.long)
        #targets = targets.to(device, dtype=torch.float)  

        ids = ids.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        token_type_ids = token_type_ids.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)        
        
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        loss = loss_fn(outputs, targets)
        
        fin_targets.append(targets.cpu().detach().numpy())
        fin_outputs.append(outputs.cpu().detach().numpy())
        
    return np.vstack(fin_outputs), np.vstack(fin_targets)

def run(rank, FLAGS):
    MAX_LEN = FLAGS['max_len']
    TRAIN_BATCH_SIZE = FLAGS['batch_size']
    VALID_BATCH_SIZE = FLAGS['batch_size']
    EPOCHS = FLAGS['epochs']    
    LR = 3e-5   
    
    dist.init_process_group(backend='nccl', world_size=FLAGS['world_size'], init_method='env://', rank=rank)
    
    dfx = pd.read_csv('train.csv').fillna("none")
    df_train, df_valid = model_selection.train_test_split(dfx, random_state=42, test_size=0.1)    
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)    
    sample = pd.read_csv('sample_submission.csv')
    target_cols = list(sample.drop('qa_id', axis=1).columns)
    train_targets = df_train[target_cols].values
    valid_targets = df_valid[target_cols].values
    
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    
    train_dataset = BERTDatasetTraining(
    qtitle = df_train.question_title.values,
    qbody = df_train.question_body.values,
    answer = df_train.answer.values,
    targets = train_targets,
    tokenizer=tokenizer,
    max_len=MAX_LEN)
    train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=FLAGS['world_size'], rank=rank, shuffle=True)	    
    
    valid_dataset = BERTDatasetTraining(
    qtitle = df_valid.question_title.values,
    qbody = df_valid.question_body.values,
    answer = df_valid.answer.values,
    targets = valid_targets,
    tokenizer=tokenizer,
    max_len=MAX_LEN)
    valid_sampler = torch.utils.data.DistributedSampler(valid_dataset, num_replicas=FLAGS['world_size'], rank=rank)	      
    
    num_train_steps = int(len(train_dataset)/TRAIN_BATCH_SIZE/FLAGS['world_size'] * EPOCHS)
    print(f'Training steps:{num_train_steps}')
    
    model = BERTBaseUncased('bert-base-uncased')
    torch.cuda.set_device(rank)
    model.cuda(rank)    
    
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=num_train_steps)
            
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    for epoch in range(EPOCHS):
        torch.save(model.state_dict(), './model_save/model.bin')
        print(f'Epoch = {epoch}')
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size = TRAIN_BATCH_SIZE,
                                                   sampler = train_sampler
                                                   )
        valid_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size = VALID_BATCH_SIZE,
                                                   sampler = valid_sampler
                                                   )        
        train_loop_fn(train_data_loader, model, optimizer, scheduler)
        o, t = eval_loop_fn(valid_data_loader, model)
        
        spear = []
        for jj in range(t.shape[1]):
            p1 = list(t[:, jj])
            p2 = list(o[:, jj])
            coef, _ = np.nan_to_num(stats.spearmanr(p1, p2))
            spear.append(coef)
        spear = np.mean(spear)
        print(f'epoch={epoch}, spearman={spear}')
        torch.save(model.state_dict(), './model_save/model.bin')


FLAGS = {'batch_size':8,
         'world_size':4,
         'epochs':3,         
         'max_len':512}
		 
if __name__ == '__main__':   
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # set to DETAIL for runtime logging.
    mp.spawn(run, nprocs=FLAGS['world_size'], args=(FLAGS,))




        