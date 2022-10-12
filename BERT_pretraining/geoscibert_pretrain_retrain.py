import os
from pathlib import Path
from tqdm import tqdm
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM, BertForMaskedLM, set_seed
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers import EarlyStoppingCallback, IntervalStrategy, SchedulerType
import math
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from argparse import ArgumentParser

import warnings
warnings.filterwarnings(action='ignore')
transformers.logging.set_verbosity_error()

'''
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('using device:', device)
'''

def ensure_dir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path

parser = ArgumentParser()
#parser.add_argument('--train_file', required=True, type=str)
#parser.add_argument('--val_file', required=True, type=str)
parser.add_argument('--model_save_dir', required=True, type=str)
parser.add_argument('--cache_dir', default=None, type=str)
args = parser.parse_args()

model_revision = 'main'
model_name = 'm3rg-iitd/matscibert'
cache_dir = ensure_dir(args.cache_dir) if args.cache_dir else None
output_dir = ensure_dir(args.model_save_dir)

#assert os.path.exists(args.train_file)
#assert os.path.exists(args.val_file)

SEED = 42
max_seq_length = 512
set_seed(SEED)

# LOAD MODEL & TOKENIZER
config_kwargs = {
    'cache_dir': cache_dir,
    'revision': model_revision,
    'use_auth_token': None,
}
config = AutoConfig.from_pretrained(model_name, **config_kwargs)

tokenizer_kwargs = {
    'cache_dir': cache_dir,
    'use_fast': True,
    'revision': model_revision,
    'use_auth_token': None,
}
tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
'''
model = BertForMaskedLM.from_pretrained(
    model_name,
    from_tf=False,
    config=config,
    cache_dir=cache_dir,
    revision=model_revision,
    use_auth_token=None,
)
'''
model = BertForMaskedLM.from_pretrained(
    '/home/ppillai6/Desktop/BERT_training/geoscibert_bkp_2/checkpoint-152771/',
    from_tf=False,
    config=config,
    cache_dir=cache_dir,
    revision=model_revision,
    use_auth_token=None,
)
model.resize_token_embeddings(len(tokenizer))

#READ DATA
geo_df = pd.read_csv('./datasets/Geo_Dataset/Training_paras_for_BERT.csv')
#geo_df = geo_df.loc[geo_df['Source']!='Onepetro']
#geo_df = geo_df.reset_index(drop=True)
print(geo_df)
# SPLIT DATA
train, val = train_test_split(geo_df, test_size=0.2, random_state=100)
print('>>TRAIN:', train.shape[0])
print('>>VAL:', val.shape[0])
train = train['Text']
val = val['Text']
train.to_csv('./datasets/Geo_Dataset/Train.csv', index=False)
val.to_csv('./datasets/Geo_Dataset/Val.csv', index=False)
# CREATE DATASET
data_files = {}
data_files["train"] = './datasets/Geo_Dataset/Train.csv'
data_files["validation"] = './datasets/Geo_Dataset/Val.csv'
extension='csv'
raw_datasets = load_dataset(extension, data_files=data_files)  

# TOKENIZE DATA

def tokenize_function(examples):
    return tokenizer(examples["Text"], padding='max_length', max_length=512)
'''
def tokenize_function(examples):
    out = tokenizer(examples["Text"], padding='max_length', max_length=512)
    out['labels'] = out['input_ids']
    return out'''

def group_texts(examples, block_size=512):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, num_proc=8, remove_columns=["Text"])

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=8,
)
#lm_datasets = tokenized_datasets

NTPU = 8
EPOCHS=1000
TRAIN_BATCHSIZE = 32
VAL_BATCHSIZE = 8
TRAIN_SIZE = len(lm_datasets["train"])
EVAL_SIZE = len(lm_datasets["validation"])
GRADACCUM = 1 #int(256/(TRAIN_BATCHSIZE * NTPU))
total_steps = TRAIN_SIZE/(TRAIN_BATCHSIZE * NTPU * GRADACCUM) * EPOCHS
print('Train size:', TRAIN_SIZE,', Eval size:',EVAL_SIZE, ', Steps:',total_steps, ', Gradient accum:', GRADACCUM)

def _mp_fn(rank):
    data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm_probability=0.15)
    training_args = TrainingArguments(
    "geoscibert",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    #evaluation_strategy = IntervalStrategy.STEPS,
    num_train_epochs=EPOCHS, #default 3
    per_device_train_batch_size=TRAIN_BATCHSIZE, #default 8
    per_device_eval_batch_size=VAL_BATCHSIZE, #default 8
    gradient_accumulation_steps=GRADACCUM, #default 1
    warmup_ratio=0,
    learning_rate=1e-5, #1e-5
    weight_decay=1e-2,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-6,
    max_grad_norm=0.0,
    push_to_hub=False,
    #logging_steps=5,
    load_best_model_at_end=True,
    tpu_num_cores=8
    #lr_scheduler_type=SchedulerType.LINEAR
    )    
    trainer = Trainer(model=model, args=training_args,train_dataset=lm_datasets["train"],eval_dataset=lm_datasets["validation"],
    #callbacks = [EarlyStoppingCallback(early_stopping_patience=3), TensorBoardCallback()],
    tokenizer=tokenizer,
    data_collator=data_collator)

    trainer.train()

xmp.spawn(_mp_fn, nprocs=8, start_method='fork')

#model.save_pretrained(output_dir)