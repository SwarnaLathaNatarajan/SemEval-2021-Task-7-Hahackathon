import torch
import pandas as pd
import argparse
import pandas
import numpy
import os
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score as accuracy_score 
from sklearn.metrics import mean_squared_error as mean_squared_error
from hahadataset import HahaDataset
from transformers import (
    XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer, )
from transformers import TrainingArguments, Trainer
from tqdm import tqdm,trange
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
import math
from IPython import embed

batch_num = 32
len_tr_inputs = 0
len_val_inputs = 0
path2spiece = 'xlnet_base_cased\spiece.model' 
tokenizer = XLNetTokenizer(vocab_file=path2spiece, do_lower_case=False)

def generate_dataloader(sentences, labels):
    global tokenizer
    #Tokenization and Segmentation
    full_input_ids = []
    full_input_masks = []
    full_segment_ids = []

    SEG_ID_A   = 0
    SEG_ID_B   = 1
    SEG_ID_CLS = 2
    SEG_ID_SEP = 3
    SEG_ID_PAD = 4

    UNK_ID = tokenizer.encode("<unk>")[0]
    CLS_ID = tokenizer.encode("<cls>")[0]
    SEP_ID = tokenizer.encode("<sep>")[0]
    MASK_ID = tokenizer.encode("<mask>")[0]
    EOD_ID = tokenizer.encode("<eod>")[0]

    for i,sentence in enumerate(sentences):
        # Tokenize sentence to token id list
        tokens_a = tokenizer.encode(sentence)
        
        # Trim the len of text
        if(len(tokens_a)>max_len-2):
            tokens_a = tokens_a[:max_len-2]
        
        tokens = []
        segment_ids = []
        
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(SEG_ID_A)
            
        # Add <sep> token 
        tokens.append(SEP_ID)
        segment_ids.append(SEG_ID_A)
        
        
        # Add <cls> token
        tokens.append(CLS_ID)
        segment_ids.append(SEG_ID_CLS)
        
        input_ids = tokens
        
        # The mask has 0 for real tokens and 1 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [0] * len(input_ids)

        # Zero-pad up to the sequence length at fornt
        if len(input_ids) < max_len:
            delta_len = max_len - len(input_ids)
            input_ids = [0] * delta_len + input_ids
            input_mask = [1] * delta_len + input_mask
            segment_ids = [SEG_ID_PAD] * delta_len + segment_ids

        assert len(input_ids) == max_len
        assert len(input_mask) == max_len
        assert len(segment_ids) == max_len
        
        full_input_ids.append(input_ids)
        full_input_masks.append(input_mask)
        full_segment_ids.append(segment_ids)

    global len_val_inputs
    global len_tr_inputs
    tr_inputs, val_inputs, tr_tags, val_tags,tr_masks, val_masks,tr_segs, val_segs = train_test_split(full_input_ids, labels,full_input_masks,full_segment_ids, random_state=4, test_size=0.2)

    tr_inputs = torch.tensor(tr_inputs)
    tr_masks = torch.tensor(tr_masks)
    tr_segs = torch.tensor(tr_segs)
    len_tr_inputs = len(tr_inputs)
    tr_tags = torch.tensor(tr_tags)
    train_data = TensorDataset(tr_inputs, tr_masks, tr_segs, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_num,drop_last=True)

    val_inputs = torch.tensor(val_inputs)
    val_masks = torch.tensor(val_masks)
    val_segs = torch.tensor(val_segs)
    val_tags = torch.tensor(val_tags)
    len_val_inputs = len(val_inputs)
    valid_data = TensorDataset(val_inputs, val_masks, val_segs, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_num)
    
    return train_dataloader, valid_dataloader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

task = 'humor_rating'

max_len = 64

data_path = r'C:\Users\krish\hamze\SemEval-2021-Task-7-Hahackathon\xlnet\data\train.csv'
df_data = pd.read_csv(data_path,sep=",",encoding="utf-8", usecols=['text', 'humor_rating'])
df_data=df_data.dropna()
train_sentences = df_data.text.to_list()
train_labels = df_data.humor_rating.to_list()
train_dataloader, valid_dataloader = generate_dataloader(train_sentences, train_labels)


tag2idx={'0': 0, '1': 1}
tag2name={tag2idx[key] : key for key in tag2idx.keys()}


epochs = 7
max_grad_norm = 1.0
# model_path = r'C:\Users\krish\hamze\SemEval-2021-Task-7-Hahackathon\xlnet\xlnet_cased_L-12_H-768_A-12'
model_path = 'xlnet-base-cased'
model = XLNetForSequenceClassification.from_pretrained(model_path, num_labels=1)
model.to(device)

# Cacluate train optimiazaion num
num_train_optimization_steps = int( math.ceil(len_tr_inputs / batch_num) / 1) * epochs
FULL_FINETUNING = True

if FULL_FINETUNING:
    # Fine tune model all layer parameters
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    # Only fine tune classifier parameters
    param_optimizer = list(model.classifier.named_parameters()) 
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

# TRAIN loop
model.train()

print("***** Running training *****")
print("  Num examples = %d"%(len_tr_inputs))
print("  Batch size = %d"%(batch_num))
print("  Num steps = %d"%(num_train_optimization_steps))
for _ in trange(epochs,desc="Epoch"):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_segs,b_labels = batch
        
        # forward pass
        outputs = model(input_ids =b_input_ids,token_type_ids=b_segs, input_mask = b_input_mask,labels=b_labels)
        loss, logits = outputs[:2]
        
        # backward pass
        loss.backward()
        
        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        
        # update parameters
        optimizer.step()
        optimizer.zero_grad()
        
    # print train loss per epoch
    print("Train loss: {}".format(tr_loss/nb_tr_steps))

xlnet_out_address = r'C:\Users\krish\hamze\SemEval-2021-Task-7-Hahackathon\xlnet\models\xlnet_out_model'

# # Save a trained model, configuration and tokenizer
# model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

# # If we save using the predefined names, we can load using `from_pretrained`
# output_model_file = os.path.join(xlnet_out_address, "pytorch_model.bin")
# output_config_file = os.path.join(xlnet_out_address, "config.json")

# # Save model into file
# torch.save(model_to_save.state_dict(), output_model_file)
# model_to_save.config.to_json_file(output_config_file)
# tokenizer.save_vocabulary(xlnet_out_address)

# model = XLNetForSequenceClassification.from_pretrained(xlnet_out_address,num_labels=len(tag2idx))

# # Set model to GPU
# model.to(device)
# Evalue loop
model.eval()

# Set acc funtion
def accuracy(out, labels):
    outputs = numpy.argmax(out, axis=1)
    return numpy.sum(outputs == labels)

eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0

y_true = []
y_predict = []
print("***** Running evaluation *****")
print("  Num examples ={}".format(len_val_inputs))
print("  Batch size = {}".format(batch_num))
for step, batch in enumerate(valid_dataloader):
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_segs,b_labels = batch
    
    with torch.no_grad():
        outputs = model(input_ids =b_input_ids,token_type_ids=b_segs, input_mask = b_input_mask,labels=b_labels)
        tmp_eval_loss, logits = outputs[:2]
    
    # Get textclassification predict result
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    tmp_eval_accuracy = accuracy(logits, label_ids)

    # Save predict and real label reuslt for analyze
    for predict in logits.tolist():
        y_predict.append(predict[0])
        
    for real_result in label_ids.tolist():
        y_true.append(real_result)

    
    eval_loss += tmp_eval_loss.mean().item()
    eval_accuracy += tmp_eval_accuracy
   
    nb_eval_steps += 1
    
    
eval_loss = eval_loss / nb_eval_steps
# eval_accuracy = eval_accuracy / len(val_inputs)
rmse= mean_squared_error(y_true, y_predict)

loss = tr_loss/nb_tr_steps 
result = {'eval_loss': eval_loss,
                  'rmse': rmse,
                  'loss': loss}
#report = classification_report(y_pred=numpy.array(y_predict),y_true=numpy.array(y_true))

# Save the report into file
output_eval_file = os.path.join(xlnet_out_address, f"eval_results_{task}.txt")
with open(output_eval_file, "w") as writer:
    print("***** Eval results *****")
    for key in sorted(result.keys()):
        print("  %s = %s"%(key, str(result[key])))
        writer.write("%s = %s\n" % (key, str(result[key])))
        


eval_path = r'C:\Users\krish\hamze\SemEval-2021-Task-7-Hahackathon\xlnet\data\public_test.csv'
df_data = pd.read_csv(eval_path, sep=",", encoding="utf-8", usecols=['id', 'text'])
sentences = df_data.text.to_list()

#Tokenization and Segmentation
full_input_ids = []
full_input_masks = []
full_segment_ids = []

SEG_ID_A   = 0
SEG_ID_B   = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4

UNK_ID = tokenizer.encode("<unk>")[0]
CLS_ID = tokenizer.encode("<cls>")[0]
SEP_ID = tokenizer.encode("<sep>")[0]
MASK_ID = tokenizer.encode("<mask>")[0]
EOD_ID = tokenizer.encode("<eod>")[0]

for i,sentence in enumerate(sentences):
    # Tokenize sentence to token id list
    tokens_a = tokenizer.encode(sentence)
    
    # Trim the len of text
    if(len(tokens_a)>max_len-2):
        tokens_a = tokens_a[:max_len-2]
    
    tokens = []
    segment_ids = []
    
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(SEG_ID_A)
        
    # Add <sep> token 
    tokens.append(SEP_ID)
    segment_ids.append(SEG_ID_A)
    
    
    # Add <cls> token
    tokens.append(CLS_ID)
    segment_ids.append(SEG_ID_CLS)
    
    input_ids = tokens
    
    # The mask has 0 for real tokens and 1 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [0] * len(input_ids)

    # Zero-pad up to the sequence length at fornt
    if len(input_ids) < max_len:
        delta_len = max_len - len(input_ids)
        input_ids = [0] * delta_len + input_ids
        input_mask = [1] * delta_len + input_mask
        segment_ids = [SEG_ID_PAD] * delta_len + segment_ids

    assert len(input_ids) == max_len
    assert len(input_mask) == max_len
    assert len(segment_ids) == max_len
    
    full_input_ids.append(input_ids)
    full_input_masks.append(input_mask)
    full_segment_ids.append(segment_ids)

input_ids = torch.tensor(full_input_ids)
input_mask = torch.tensor(full_input_masks)
segs = torch.tensor(full_segment_ids)

test_data = TensorDataset(input_ids, input_mask, segs)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_num)

y_true = []
y_predict = []

print("***** Running Testing *****")
for step, batch in enumerate(test_dataloader):
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_segs = batch
    
    with torch.no_grad():
        outputs = model(input_ids=b_input_ids, token_type_ids=b_segs, input_mask=b_input_mask)
        logits = outputs[:1]
  
    # Get textclassification predict result
    outputs = outputs[0].cpu().detach()
    
    # Save predict and real label reuslt for analyze
    for predict in outputs.tolist():
        y_predict.append(predict[0])
        

output_list = []
for pred in y_predict:
    temp = {}
    temp['humor_rating'] = pred
    output_list.append(temp)

out_df = pandas.DataFrame(output_list)
out_df.to_csv('sub_xlnet_humor_rating.csv', index_label='id')
print("** Generated sub_xlnet_humor_rating.csv **")