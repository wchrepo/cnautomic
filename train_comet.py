#%% load libraries

# from datasets import load_dataset,load_metric
# from data import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, BartTokenizerFast, AdamW, AddedToken,
    T5ForConditionalGeneration, T5TokenizerFast,
    get_linear_schedule_with_warmup
)
import torch
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
import os,time
import argparse
import sys
import json
import mlflow
import random
from tqdm.auto import tqdm
from triple_utils import random_naming,example2text
#%% argparse
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=None, type=int,
                    help='node rank for distributed training')
parser.add_argument('--config', default=None, type=str,
                    help='config file path')
#cfg不再记录ddp信息。
# parser.add_argument('--ignore_worldsize_mismatch', action='store_true',
#                     help='if load a config that has set worldsize, which is however different from the actual worldsize in this run, this flag will make the script ignore worldsize in config.')
parser.add_argument('--dry_run', action='store_true',
                    help='run without training (maybe you just want caching prepared data)')
parser.add_argument('--force_seed',default=None,type=int,help='set the random seed (ignoring the seed in config file)')
parser.add_argument("--dataset_name",default="atomic-zh",type=str)
if sys.argv[0].split('/')[-1].startswith("ipykernel_launcher"):
    args = parser.parse_args([])
else:
    args = parser.parse_args()
local_rank = args.local_rank
#%%
cfg = {}
old_vars = set()
old_vars.update(k for k in globals() if not k.startswith('_'))
#%% some hyper-parameters
underlying_model_name = "google/mt5-base"
learning_rate = 1e-04
epochs = 1
iterations = 0 #0
cycle = 500
warm_up_steps = 0#0.002*iterations
weight_decay = 0.0
batch_size = 32#32
seed = 42
#Cut down memory usage by accumulating tiny steps for multiple backwards;
#Should be divided exactly by batch_size
accumulation_tiny_steps = 1 
shuffle = True
shuffle_evaluation=False
validation_size = 1000
validation_num_generation = 50
# generation_params = {
#     "max_length":50,
#     "early_stopping":True,
#     "num_beams":5,
#     "num_return_sequences":5,
# }
generation_params = {
    "max_length":50,
    "early_stopping":True,
}
ddp = args.local_rank is not None
device = 'cuda'
log_dir = 'comet_model/'
save_best = True #
truncation = True
max_query_length = 48
max_target_length = 24 
use_special_rel_tokens = True
add_placeholder_in_special_tokens = False #no_placeholder/mask_placeholder时不应该把placeholder加入specialtokens
placeholder_token = "<extra_id_0>" # "no_placeholder","mask_placeholder","<gen>"
# placeholder_token = "" #先不加了，之后不行再试试extra_id_0这一套
scheduler_cycling = False
scheduler_cycling_strategy = "epoch"
save_model_per_epoch = True
#%% dataset 
dataset_name = args.dataset_name
if dataset_name == 'atomic-zh':
    dataset_path = 'data/atomic-zh'
    test_path = "data/atomic-zh/sample_test_atomic_zh.json"
    name_mapping = {
        "<某人X>":"他",
        "<某人Y>":"别人",
        "<某人Z>":"其他人",
        "<某样东西Y>":"某样东西",
    }
else:
    # dataset_name = 'cnautomic-small'
    dataset_path = 'results/filtered_triples/1/before_iteration'
    test_path = "data/atomic-zh/sample_test_atomic_zh_cnautomic.json"
    # name_mapping = {
    #     "<某人X>":"<某人X>",
    #     "<某人Y>":"<某人X>",
    #     "<某人Z>":"<某人X>",
    #     "<某样东西Y>":"<某样东西Y>",
    # }
    name_mapping = {
        "<某人X>":"某人X",
        "<某人Y>":"某人Y",
        "<某人Z>":"某人Z",
        "<某样东西Y>":"某样东西",
    }
relation_token_mappings = {
    "HinderedBy":"<HinderedBy>",
    "xAttr":"<xAttr>",
    "xEffect":"<xEffect>",
    "xIntent":"<xIntent>",
    "xNeed":"<xNeed>",
    "xReact":"<xReact>",
    "xWant":"<xWant>",
}
if not use_special_rel_tokens:
    rel_templates = {
        "xWant":"{head_item}，在此之后，{X}想要{tail_item}",
        "xAttr":"{head_item}，据此，可以看出{X}是{tail_item}",
        "xEffect":"{head_item}。结果，{X}{tail_item}",
        "xReact":"{head_item}，对此，{X}感觉{tail_item}",
        "xNeed":"{head_item}，在此之前，{X}需要{tail_item}",
        "xIntent":"{head_item}，{X}的意图是{tail_item}",
        "HinderedBy":"{head_item}，这受到阻碍，因为{tail_item}"
    }
    
# dataset_name = 'atomic-zh'
# dataset_path = 'data/atomic-zh'
# test_path = "data/atomic-zh/sample_test_atomic_zh_cnautomic.json"

exp_name = f'{dataset_name}_{"rel_token" if use_special_rel_tokens else "rel_prompt"}_{"extra_id_placeholder" if placeholder_token else "no_placeholder"}'
#%%
new_vars = set(k for k in globals() if not k.startswith('_'))
cfg_vars = new_vars-old_vars
cfg = {k:v for k,v in globals().items() if k in cfg_vars }
load_config_path = args.config
if load_config_path: #载入config
    with open(load_config_path) as f:
        cfg.update(json.load(f))

        globals().update(cfg)

#%% dpp initialize
is_main_process = (not ddp or local_rank == 0) 
if ddp:
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    print("launch process",local_rank)
    world_size = torch.distributed.get_world_size()


#%% tokenizer & model

tokenizer = AutoTokenizer.from_pretrained(underlying_model_name)
# tokenizer = BartTokenizerFast.from_pretrained(underlying_model_name,add_prefix_space=True,padding_side='left')
model = AutoModelForSeq2SeqLM.from_pretrained(underlying_model_name)
# model = BartForConditionalGeneration.from_pretrained(underlying_model_name,force_bos_token_to_be_generated=True)

#%%collate function
def collate_fn_for_flattened(batch):
    queries,responses = zip(*batch)
    if truncation:
        new_batch = tokenizer(list(queries),return_tensors='pt',padding='longest',truncation=truncation,max_length = max_query_length)
    else:
        new_batch = tokenizer(list(queries),return_tensors='pt',padding='longest')
    with tokenizer.as_target_tokenizer():
        if truncation:
            outputs = tokenizer(list(responses),return_tensors='pt',padding='longest',truncation=truncation,max_length = max_target_length)
        else:
            outputs = tokenizer(list(responses),return_tensors='pt',padding='longest')
        labels = outputs['input_ids']
        labels[labels==tokenizer.pad_token_id] = -100
        new_batch['labels']=labels
    return new_batch


#%% load atomic data
# dataset = load_dataset('atomic')


import csv
def load_atomic2020_dataset(path):
    splits = ['train','dev','test']
    data = {}
    for split in splits:
        data[split] = []
        with open(os.path.join(path,f"{split}.tsv")) as f:
            source_reader = csv.reader(f,dialect='excel-tab')
            for line in source_reader:
                # line = line.rstrip('\n').split('\t')
                #can do some normalize here
                if line[0] and line[1] and line[2]: 
                    h,r,t = line
                    if t and r in relation_token_mappings:
                        d = {"h":h,"r":r,"t":t}
                        data[split].append(d)
    return data
if dataset_name == 'atomic-zh':
    dataset = load_atomic2020_dataset(dataset_path)
    random.seed(1)
    random.shuffle(dataset['dev'])
else:
    def apply_name_mapping(triple):
        h,r,t=triple['h'],triple['r'],triple['t']
        for k,v in name_mapping.items():
            h = h.replace(k,v)
            t = t.replace(k,v)
        return {'h':h,'r':r,'t':t}
    dataset = {}
    subset_name = dataset_name.split('-')[-1]
    all_data = []
    with open(os.path.join(dataset_path,f"{subset_name}_filtered.jsonl.txt")) as f:
        for line in f:
            d = json.loads(line.rstrip())

            triple = apply_name_mapping(d['triple'])
            all_data.append(triple) 
    random.seed(1)
    random.shuffle(all_data)
    dataset['train'] = all_data[:int(len(all_data)*0.9)]
    dataset['dev'] = all_data[int(len(all_data)*0.9):]


if underlying_model_name.split('/')[-1].startswith(('mt5','t5')):
    def process_tail(t):
        if placeholder_token.startswith("<extra_id_0>"):
            # return f" <extra_id_0>{t} <extra_id_1>"
            return f" <extra_id_0>{t}"
        else:
            return t

print("building query responses")
atomic_query_responses = {}
for split_name,split_data in dataset.items():
    atomic_query_responses[split_name] = {}
    for d in split_data:
        if use_special_rel_tokens:
            rel_token = relation_token_mappings[d['r']]
            query = f"{d['h']} {rel_token} {placeholder_token}".strip()
        else:
            query = example2text({'h':d['h'],'r':d['r'],'t':placeholder_token},name_mapping,rel_templates[d['r']])
        if query not in atomic_query_responses[split_name]:
            atomic_query_responses[split_name][query] = []
        atomic_query_responses[split_name][query].append(process_tail(d['t']))
print("building flattened pairs")
atomic_flattened = {}
for split_name,queries_responses in atomic_query_responses.items():
    atomic_flattened[split_name] = []
    for query,responses in queries_responses.items():
        for response in responses:
            atomic_flattened[split_name].append((query,response))
data_for_loader = atomic_flattened
collate_fn = collate_fn_for_flattened

#%% reproducibility
if args.force_seed is not None:
    seed = args.force_seed
    cfg['seed']=seed
import torch
torch.manual_seed(seed)
import numpy as np
np.random.seed(seed)
import random
random.seed(seed)
#%% add new tokens
# added_tokens = list(atomic_relation_mappings.values()) + [gen_token]
if use_special_rel_tokens:
    added_tokens = [ 
        AddedToken(token,lstrip=True,
            rstrip=False)
        for token in 
            list(relation_token_mappings.values())
            + ([placeholder_token] if add_placeholder_in_special_tokens else [])
    ]
    tokenizer.add_special_tokens({"additional_special_tokens":added_tokens})
    model.resize_token_embeddings(len(tokenizer))

#%% dataloader and  parallel
node_batch_size = batch_size//accumulation_tiny_steps
train_sampler = None
if shuffle:
    train_sampler = torch.utils.data.RandomSampler(data_for_loader['train'])
if ddp:
    assert node_batch_size%world_size == 0
    node_batch_size = node_batch_size//world_size
    train_sampler = torch.utils.data.DistributedSampler(data_for_loader['train'],shuffle=shuffle)
train_dataloader = torch.utils.data.DataLoader(data_for_loader['train'],
    batch_size=node_batch_size,sampler=train_sampler,
    collate_fn=collate_fn)
if is_main_process:
    dev_dataloader = torch.utils.data.DataLoader(data_for_loader['dev'],
        batch_size=node_batch_size,shuffle=shuffle_evaluation,
        collate_fn=collate_fn)

if args.dry_run:
    assert (False,"dry run end")
# %% prepare for training
model_name = os.path.join(dataset_name,underlying_model_name.split('/')[-1],f"{exp_name}_{learning_rate}_e{epochs}_i{iterations}_s{seed}_b{batch_size}_wd{weight_decay}_ws{warm_up_steps}_"
    f"{time.strftime('%Y%m%d_%a_%H:%M:%S')}")
if epochs != 0:
    assert iterations==0
    iterations = epochs * ((len(train_dataloader.dataset)-1)// batch_size + 1)
if is_main_process:
    serialization_dir = os.path.join(log_dir,model_name)
    mlflow.set_experiment("comet_model")
    mlflow.start_run(run_name=model_name.split('/')[-1])
    mlflow.set_tags({
        "dataset_path":dataset_path,
        "dataset_name":dataset_name,
        "model":underlying_model_name,
        "output_path":serialization_dir,
        "intent":"",
    })
    mlflow.log_params({k:v for k,v in cfg.items() if len(str(v))<250})
    tokenizer.save_pretrained(serialization_dir)

    with open(os.path.join(serialization_dir,'exp_config.json'),'w') as f:
        json.dump(cfg,f,ensure_ascii=False,indent=4)
model = model.to(device=device)
model_ = model #for generation
if ddp:
    model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank])
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters,lr=learning_rate,eps=1e-8)
if not scheduler_cycling:
    scheduler = get_linear_schedule_with_warmup(optimizer,warm_up_steps,iterations)
else:
    if scheduler_cycling_strategy == "epoch":
        scheduler_cycling_step = (len(train_dataloader.dataset)// batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer,warm_up_steps,scheduler_cycling_step)
step = 0
best_dev_loss = 1e10
#%%
train_iter = iter(train_dataloader)
if is_main_process:
    pbar = tqdm(total=iterations,dynamic_ncols=True)
    epoch_counter = 0
while step <= iterations:
    if is_main_process and (step % cycle == 0 and step > 0): #validation
        with torch.no_grad():
            model.eval()
            pbar.set_description('validating...')
            dev_allset_micro_loss = 0.
            dev_token_loss = 0.
            dev_token_count = 0
            dev_sample_loss = 0. #avg on sample
            dev_sample_count = 0
            for batch in tqdm(dev_dataloader,desc=f'validating ...',leave=False):
                if dev_sample_count>=validation_size:
                    break
                # if train_mode == 'one2multi' and mode_settings["adaptive_order"]:
                #     batch = prepare_batch_for_one2multi(model_,batch)
                batch = {k:v.to(device=device) for k,v in batch.items()}
                result = model(**batch)
                loss = torch.nn.functional.cross_entropy(
                    result['logits'].reshape(-1,result['logits'].size(2)),
                    batch['labels'].reshape(-1,),
                    reduction='none'
                ).reshape(result['logits'].size(0),-1)
                labels_mask = (batch['labels'] != -100) 
                dev_token_loss += loss.sum().item()
                dev_token_count += labels_mask.sum().item()
                dev_sample_loss += (loss.sum(dim=-1)/labels_mask.sum(dim=-1)).sum().item()
                dev_sample_count += result['logits'].size(0)
                del result
                del loss
                del labels_mask
            dev_micro_avg_loss = dev_token_loss/dev_token_count
            dev_macro_avg_loss = dev_sample_loss/dev_sample_count
            mlflow.log_metric('dev/micro_avg_loss',dev_micro_avg_loss,step)
            # sw.add_scalar('dev/macro_avg_loss',dev_macro_avg_loss,step)
            if dev_micro_avg_loss < best_dev_loss or not save_best:
                best_dev_loss = dev_micro_avg_loss
                model_.save_pretrained(serialization_dir)
            gts = {}
            res = {}
            generation_results = \
            "|Queries|Generation Results|\n"\
            "|-|-|\n"
            for i,(query,responses) in enumerate(tqdm(atomic_query_responses['dev'].items())):
                if i==validation_num_generation:
                    break
                else:
                    results = tokenizer.batch_decode(
                        model_.generate(**tokenizer(query,return_tensors='pt').to(device=device),**generation_params),
                        skip_special_tokens=True
                    )
                    res[query]= [results[0]]
                    gts[query]= responses
                generation_results+=f"|`{query}`|`{str(results)}`|\n"

            # score_dict,_ = eval_scorers.evaluate(gts,res)
            # score_text = ""
            # for key,score in score_dict.items():
            #     score_text += f"- **{key}**: {score} \n"
            mlflow.log_text(generation_results,f"generation_{step}.md")
    model.train()
    optimizer.zero_grad()
    batch_loss = torch.tensor(0.,device=device)
    for tiny_step in range(accumulation_tiny_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)
            if scheduler_cycling:
                scheduler = get_linear_schedule_with_warmup(optimizer,warm_up_steps,scheduler_cycling_step)
            if is_main_process and save_model_per_epoch:
                epoch_counter += 1
                tokenizer.save_pretrained(os.path.join(serialization_dir,f"epoch_{epoch_counter}"))
                model_.save_pretrained(os.path.join(serialization_dir,f"epoch_{epoch_counter}"))
                with open(os.path.join(serialization_dir,f"epoch_{epoch_counter}",'exp_config.json'),'w') as f:
                    json.dump(cfg,f,ensure_ascii=False,indent=4)
        # if train_mode == 'one2multi' and mode_settings["adaptive_order"]:
        #     batch = prepare_batch_for_one2multi(model_,batch)
        batch = {k:v.to(device=device) for k,v in batch.items()}
        result = model(**batch)
        loss = result['loss']/accumulation_tiny_steps
        loss.backward()
        batch_loss += loss.item()
    optimizer.step()
    scheduler.step()
    step+=1
    if ddp:
        # loss = loss.detach()
        losses = [torch.zeros_like(batch_loss) for i in range(world_size)]
        torch.distributed.all_gather(tensor_list=losses,tensor=batch_loss)
        batch_loss = torch.stack(losses).mean()
    if is_main_process:
        pbar.set_description('training...')
        pbar.update()
        mlflow.log_metric('train/loss',batch_loss.item(),step)
    del result
    del loss
if is_main_process:
    pbar.close()
    #save final model
    tokenizer.save_pretrained(os.path.join(serialization_dir,f"final"))
    model_.save_pretrained(os.path.join(serialization_dir,f"final"))
    with open(os.path.join(serialization_dir,f"final",'exp_config.json'),'w') as f:
        json.dump(cfg,f,ensure_ascii=False,indent=4)
#%% test
tokenizer = AutoTokenizer.from_pretrained(serialization_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(serialization_dir)
model.eval()
def post_process(r):
    r = r.removeprefix("<extra_id_0>")
    if "<extra_id_" in r:
        r = r[:r.index("<extra_id_")]
    return r
with open(test_path) as f:
    test_data = json.load(f)
test_results = []
for d in test_data:
    if use_special_rel_tokens:
        rel_token = relation_token_mappings[d['r']]
        query = f"{d['h']} {rel_token} {placeholder_token}".strip()
    else:
        #注意由于已经提前替换了name,所以这里不需要再对h、t做处理
        query = example2text({'h':d['h'],'r':d['r'],'t':placeholder_token},name_mapping,rel_templates[d['r']])
    results = tokenizer.batch_decode(
        model_.generate(**tokenizer(query,return_tensors='pt').to(device=device),**generation_params),
        skip_special_tokens=True
    )
    results = [post_process(r) for r in results]
    test_results.append({"query":query,"responses":results})
with open(os.path.join(serialization_dir,"test_results.json"),'w') as f:
    json.dump(test_results,f,ensure_ascii=False,indent=2)

