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



def metrics_acc(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


def metrics_rmse(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions
    rmse = mean_squared_error(labels, preds)
    rmse = numpy.float64(rmse)
    return {"rmse": rmse}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_from_checkpoint', type=str)
    parser.add_argument('--continue_training', type=str)
    parser.add_argument('--output_directory', type=str)
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_gpus', type=int, default=4)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    task = 'is_humor'
    path2spiece = 'xlnet_base_cased\spiece.model' 
    max_len = 64
    tokenizer = XLNetTokenizer(vocab_file=path2spiece, do_lower_case=False)
    data_path = r'C:\Users\krish\hamze\SemEval-2021-Task-7-Hahackathon\xlnet\data\train.csv'
    df_data = pd.read_csv(data_path,sep=",",encoding="utf-8", usecols=['text', 'is_humor'])
    print(df_data.columns)
    print(df_data.head(n=20))
    print(df_data.is_humor.unique())
    print(df_data.is_humor.value_counts())
    sentences = df_data.text.to_list()
    labels = df_data.is_humor.to_list()
    print(sentences[0], labels[0])
    tag2idx={'0': 0, '1': 1}
    tag2name={tag2idx[key] : key for key in tag2idx.keys()}


    
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

    # if 3 > i:
    #     print("No.:%d"%(i))
    #     print("Sentence: %s"%(sentence))
    #     print("input_ids:%s"%(input_ids))
    #     print("attention_masks:%s"%(input_mask))
    #     print("segment_ids:%s"%(segment_ids))
    #     print("\n")
    
tags = [tag2idx[str(lab)] for lab in labels]
tr_inputs, val_inputs, tr_tags, val_tags,tr_masks, val_masks,tr_segs, val_segs = train_test_split(full_input_ids, tags,full_input_masks,full_segment_ids, random_state=4, test_size=0.3)

# print(len(tr_inputs),len(val_inputs),len(tr_segs),len(val_segs))

tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)
tr_segs = torch.tensor(tr_segs)
val_segs = torch.tensor(val_segs)

batch_num = 32
train_data = TensorDataset(tr_inputs, tr_masks,tr_segs, tr_tags)
train_sampler = RandomSampler(train_data)
# Drop last can make batch training better for the last one
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_num,drop_last=True)

valid_data = TensorDataset(val_inputs, val_masks,val_segs, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_num)

# model_path = r'C:\Users\krish\hamze\SemEval-2021-Task-7-Hahackathon\xlnet\xlnet_cased_L-12_H-768_A-12'
model_path = 'xlnet-base-cased'
model = XLNetForSequenceClassification.from_pretrained(model_path, num_labels=len(tag2idx))
# print(model )
model.to(device)

epochs = 7
max_grad_norm = 1.0
# Cacluate train optimiazaion num
num_train_optimization_steps = int( math.ceil(len(tr_inputs) / batch_num) / 1) * epochs
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
print("  Num examples = %d"%(len(tr_inputs)))
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
print("  Num examples ={}".format(len(val_inputs)))
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
#     print(tmp_eval_accuracy)
#     print(np.argmax(logits, axis=1))
#     print(label_ids)
    
    # Save predict and real label reuslt for analyze
    for predict in numpy.argmax(logits, axis=1):
        y_predict.append(predict)
        
    for real_result in label_ids.tolist():
        y_true.append(real_result)

    
    eval_loss += tmp_eval_loss.mean().item()
    eval_accuracy += tmp_eval_accuracy
   
    nb_eval_steps += 1
    
    
eval_loss = eval_loss / nb_eval_steps
eval_accuracy = eval_accuracy / len(val_inputs)
loss = tr_loss/nb_tr_steps 
result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'loss': loss}
report = classification_report(y_pred=numpy.array(y_predict),y_true=numpy.array(y_true))

# Save the report into file
output_eval_file = os.path.join(xlnet_out_address, f"eval_results_{task}.txt")
with open(output_eval_file, "w") as writer:
    print("***** Eval results *****")
    for key in sorted(result.keys()):
        print("  %s = %s"%(key, str(result[key])))
        writer.write("%s = %s\n" % (key, str(result[key])))
        
    print(report)
    writer.write("\n\n")  
    writer.write(report)

    # train_dataset = HahaDataset(
    #     input_file='data\train.csv',
    #     tokenizer=tokenizer, 
    #     max_len=64, 
    #     task=task, 
    #     split='train')
    # print(train_dataset.labels.value_counts())
    # eval_dataset = HahaDataset(
    #     input_file='data/train.csv', 
    #     tokenizer=tokenizer, 
    #     max_len=64, 
    #     task=task, 
    #     split='eval')
    # test_dataset = HahaDataset(
    #     input_file='data/public_test.csv', 
    #     tokenizer=tokenizer, 
    #     max_len=64, 
    #     task=task, 
    #     split='test',
    # )

    #model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

    #warmup_steps = int(args.max_steps * .01)

    #training_args = TrainingArguments(output_dir=args.output_directory, max_steps=args.max_steps, per_device_train_batch_size=args.batch_size, logging_steps=25, save_total_limit=1, evaluate_during_training=True, eval_steps=50, learning_rate=2e-5, warmup_steps=warmup_steps, load_best_model_at_end=True, metric_for_best_model='eval_accuracy', disable_tqdm=True)

    #trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, compute_metrics=metrics_acc, )

    #trainer.train()

    #predictions = trainer.predict(test_dataset)
    #is_humor_preds = predictions.predictions.argmax(-1)

    #print(predictions)
    #print(is_humor_preds)

    #output_list = []
    #for pred in is_humor_preds:
    #    temp = {}
    #    temp['is_humor'] = pred
    #    output_list.append(temp)

    #out_df = pandas.DataFrame(output_list)
    #out_df.to_csv('submission_is_humor.csv', index_label='id')




