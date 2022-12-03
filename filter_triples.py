#%%
from transformers import (
    AutoTokenizer,AutoModelForSequenceClassification, XLMRobertaForSequenceClassification, XLMRobertaTokenizerFast, AdamW, AddedToken,
    T5ForConditionalGeneration, T5TokenizerFast,
    get_linear_schedule_with_warmup,get_constant_schedule_with_warmup
)
import torch
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
import os,time
import argparse
import sys
import json
import mlflow
from sklearn import metrics
from tqdm.auto import tqdm
import numpy as np
from triple_utils import random_naming,example2text
#%%
parser = argparse.ArgumentParser()
parser.add_argument('path',type=str)
parser.add_argument('--head_filter_path', type=str) 
parser.add_argument('--tail_filter_path', type=str) 
parser.add_argument('--triple_filter_path', type=str) 
parser.add_argument('--head_filter_device', type=str) 
parser.add_argument('--tail_filter_device', type=str) 
parser.add_argument('--triple_filter_device', type=str) 
args=parser.parse_args()
#%%
#构造一个wrapper，包含模型的各种功能。能够接收一系列三元组，并且返回打分。构造wrapper只需要提供模型的path
class CriticWrapper:
    def __init__(self,load_path,device="cuda",batch_size=128) -> None:
        self.load_path = load_path
        with open(os.path.join(load_path,"exp_config.json")) as f:
            self.exp_config = json.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path,add_prefix_space=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        self.model.to(device=device)
        self.model.eval()
        self.batch_size = batch_size
        if "critic_templates" in self.exp_config:
            self.critic_templates = self.exp_config['critic_templates']
        elif self.exp_config['task'] == "head":
            self.critic_templates = {
                "xWant":"{head_item}",
                "xAttr":"{head_item}",
                "xEffect":"{head_item}",
                "xReact":"{head_item}",
                "xNeed":"{head_item}",
                "xIntent":"{head_item}",
                "HinderedBy":"{head_item}"
            }
        elif self.exp_config['task'] == "tail":
            self.critic_templates = {
                "xWant":"{tail_item}",
                "xAttr":"{tail_item}",
                "xEffect":"{tail_item}",
                "xReact":"{tail_item}",
                "xNeed":"{tail_item}",
                "xIntent":"{tail_item}",
                "HinderedBy":"{tail_item}"
            }
        elif self.exp_config['task'] == "head_rel":
            self.critic_templates = {
                "xWant":"{head_item}##SEP##在此之后，{X}想要<mask>",
                "xAttr":"{head_item}##SEP##据此，可以看出{X}是<mask>",
                "xEffect":"{head_item}##SEP##结果，{X}<mask>",
                "xReact":"{head_item}##SEP##对此，{X}感觉<mask>",
                "xNeed":"{head_item}##SEP##在此之前，{X}需要<mask>",
                "xIntent":"{head_item}##SEP##{X}的意图是<mask>",
                "HinderedBy":"{head_item}##SEP##这受到阻碍，因为<mask>"
            }
        elif self.exp_config['task'] == "tail_rel":
            self.critic_templates = {
                "xWant":"在此之后，{X}想要{tail_item}",
                "xAttr":"据此，可以看出{X}是{tail_item}",
                "xEffect":"结果，{X}{tail_item}",
                "xReact":"对此，{X}感觉{tail_item}",
                "xNeed":"在此之前，{X}需要{tail_item}",
                "xIntent":"{X}的意图是{tail_item}",
                "HinderedBy":"这受到阻碍，因为{tail_item}"
            }
        else:
            self.critic_templates = {
                "xWant":"{head_item}##SEP##在此之后，{X}想要{tail_item}",
                "xAttr":"{head_item}##SEP##据此，可以看出{X}是{tail_item}",
                "xEffect":"{head_item}##SEP##结果，{X}{tail_item}",
                "xReact":"{head_item}##SEP##对此，{X}感觉{tail_item}",
                "xNeed":"{head_item}##SEP##在此之前，{X}需要{tail_item}",
                "xIntent":"{head_item}##SEP##{X}的意图是{tail_item}",
                "HinderedBy":"{head_item}##SEP##这受到阻碍，因为{tail_item}"
            }
        # with open(os.path.join(self.exp_config['dataset_path'],"config.json")) as f:
        #     self.triple_config = json.load(f)
    def triple_to_sequence(self,triple):
        if self.exp_config['naming_setting'] == "random":
            #TODO 暂未实现根据关系调整参数的功能
            name_mapping = random_naming(
                base_name_mapping=self.exp_config["base_name_mapping"],
                names_pool=self.exp_config["names_pool"]
            ) #或许可以用names-dataset pip安装
        elif self.exp_config['naming_setting'] == "base":
            name_mapping = self.exp_config["base_name_mapping"]
        else:
            name_mapping = {k:k for k in self.exp_config["base_name_mapping"]}
        sequence = example2text(triple,name_mapping=name_mapping,
            example_prompt_template=self.critic_templates[triple['r']])
        sequence = sequence.replace("<mask>",self.tokenizer.mask_token)
        sequence_pair = sequence.split("##SEP##")
        if len(sequence_pair)==1:
            return sequence_pair[0]
        else:
            return tuple(sequence_pair) 

    def prepare_input(self,triples):
        sequences = [self.triple_to_sequence(triple) for triple in triples]
        new_batch = self.tokenizer(list(sequences),return_tensors='pt',padding='longest')
        return new_batch
    
    def infer(self,list_of_triples):
        all_probs = []
        for i in tqdm(range(0,len(list_of_triples),self.batch_size)):
            batch_triples = list_of_triples[i:i+self.batch_size]
            probs_list = []
            if self.exp_config['naming_setting'] == "random":
                repeat_num = 5
            else:
                repeat_num = 1
            for j in range(repeat_num):
                inputs = self.prepare_input(batch_triples)
                out = self.model(**inputs.to(device=self.model.device))
                probs = out.logits.softmax(-1)[:,1].tolist()
                probs_list.append(probs)
            probs = np.array(probs_list).mean(axis=0)
            all_probs.extend(probs.tolist())
        return all_probs
        




head_filter = CriticWrapper(
    args.head_filter_path,
    device=args.head_filter_device
)

triple_filter = CriticWrapper(
    args.triple_filter_path,
    device=args.triple_filter_device
)
tail_filter = CriticWrapper(
    args.tail_filter_path,
    device=args.tail_filter_device
)

#%%



#%%
head_threshold = 0.8
tail_threshold = 0.35
#确定triple thresholds的方法：应该依靠precision.
# 但是实际上precision的估计并不准确，并不是单调递减的。
# 我们应该检查所有检查点，找到最后一个比预定大小大的，如果没有，就用记录的最大的。


triple_thresholds_settings = {
    "s":{
        "xWant":0.60,
        "xAttr":0.50,
        "xEffect":0.45,
        "xReact":0.75,
        "xNeed":0.45,
        "xIntent":0.65,
        "HinderedBy":0.7
    },
    "b":{
        "xWant":0.4,
        "xAttr":0.5,
        "xEffect":0.35,
        "xReact":0.65,
        "xNeed":0.4,
        "xIntent":0.6,
        "HinderedBy":0.65
    },
    "l":{
        "xWant":0.30,
        "xAttr":0.35,
        "xEffect":0.25,
        "xReact":0.50,
        "xNeed":0.3,
        "xIntent":0.55,
        "HinderedBy":0.55
    }
}



#%%
with open(os.path.join(args.path,"normalized.jsonl.txt")) as f:
    to_pred_list = []
    for line in f:
        to_pred_list.append(json.loads(line.strip()))
triples = [d['triple'] for d in to_pred_list]
#%%
head_scores = head_filter.infer(triples)
tail_scores = tail_filter.infer(triples)
triple_scores = triple_filter.infer(triples)
# %%
for d,head_score,tail_score,triple_score in zip(
    to_pred_list,head_scores,tail_scores,triple_scores):
    d['scores'] = {}
    d['scores']['head'] = head_score
    d['scores']['tail'] = tail_score
    d['scores']['triple']=triple_score
    d['subset'] = []
    for name,triple_thresholds in triple_thresholds_settings.items():
        if (d['scores']['head']>=head_threshold
            and d['scores']['tail']>=tail_threshold
            and d['scores']['triple'] >= triple_thresholds[d['triple']['r']]
        ):
            d['subset'].append(name)
with open(os.path.join(args.path,"all_scored.jsonl.txt"),'w') as f:
    for d in to_pred_list:
        print(json.dumps(d,ensure_ascii=False),file=f)
#%%

for name,triple_thresholds in triple_thresholds_settings.items():
    with open(os.path.join(args.path,f"{name}_filtered.jsonl.txt"),'w') as f:
        for d in to_pred_list:
            if name in d['subset']:
                print(json.dumps(d,ensure_ascii=False),file=f)
# %%
