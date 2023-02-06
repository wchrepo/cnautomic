#本代码功能：处理atomic-zh(https://github.com/XiaoMi/C3KG)，按atomic原数据集划分对atomic-zh进行划分
#%%
import os
import csv
import json

#%%
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
                if line[0] and line[1] and line[2]: #
                    h,r,t = line
                    #normalization, may influence the results
                    if t:
                        d = {"h":h,"r":r,"t":t}
                        data[split].append(d)
    return data

atomic2020_data = load_atomic2020_dataset("data/atomic2020")

atomic2020_heads = {}
for split,data in atomic2020_data.items():
    atomic2020_heads[split] = set()
    for d in data:
        atomic2020_heads[split].add(d['h'])
#%%
# all_atomic_zh_data = []
# with open("data/atomic-zh/ATOMIC_Chinese.tsv") as f:
#     f.readline()
#     source_reader = csv.reader(f,dialect='excel-tab')
#     for line in source_reader:
#         if line[0] and line[1] and line[2]:
#             h,r,t = line
#             if t:
#                 all_atomic_zh_data.append({"h":h,"r":r,"t":t})
atomic_zh_data = load_atomic2020_dataset("data/atomic-zh")
# %% 构造映射
# %%
en2cn = {}
with open("data/atomic-zh/head_shortSentence.csv") as f:
    source_reader = csv.DictReader(f)
    for row in source_reader:
        en2cn[row['head']] = row['head_translated']
with open("data/atomic-zh/head_phrase.csv") as f:
    source_reader = csv.DictReader(f)
    for row in source_reader:
        en2cn[row['head']] = row['head_translated']
# %%
#有人-> PersonX
#别人,其他人-> PersonY

#从atomic2020的test里采样1000个。
#首先需要保留我们需要的关系
# 按head,rel进行分组，然后采样1000个head rel组合

kept_relations = ["xWant","xAttr","xIntent","xEffect","xReact","xNeed","HinderedBy"]
to_be_grouped = []
for d in atomic2020_data['test']:
    if d['r'] in kept_relations:
        to_be_grouped.append(d)

atomic_grouped = {} #(h,r)->t
for d in to_be_grouped:
    atomic_grouped[(d['h'],d['r'])] = d['t']

atomic_zh_to_be_grouped = []
for d in atomic_zh_data['test']:
    if d['r'] in kept_relations:
        atomic_zh_to_be_grouped.append(d)

atomic_zh_grouped = {} #(h,r)->t
for d in atomic_zh_to_be_grouped:
    atomic_zh_grouped[(d['h'],d['r'])] = d['t']

import random
random.seed(1)

zh_grouped_keys = list(atomic_zh_grouped.keys())
zh_sample_keys = random.sample(zh_grouped_keys,1000)
zh_sample_groups = {key:atomic_zh_grouped[key] for key in zh_sample_keys}

# %%
import re
def convert_atomic_zh_phrase_to_automic(p):
    p = re.sub(r"某人|有人","PersonX",p,1)
    p = re.sub(r"某人|有人","PersonY",p,1)
    p = re.sub(r"别人|其他人","PersonY",p)
    p = p.replace("PersonX","<某人X>")
    p = p.replace("PersonY","<某人Y>")
    return p

sample_groups_cnautomic = {(convert_atomic_zh_phrase_to_automic(key[0]),key[1]):\
    [convert_atomic_zh_phrase_to_automic(v) for v in value] for key,value in zh_sample_groups.items()}
#%%

import random
random.seed(1)

zh_sample_triples = random.sample(atomic_zh_to_be_grouped,1000)
zh_sample_triples_cnautomic = [{'h':convert_atomic_zh_phrase_to_automic(d['h']),'r':d['r'],'t':convert_atomic_zh_phrase_to_automic(d['t'])} for d in zh_sample_triples]
# %%
with open("data/atomic-zh/sample_test_atomic_zh.json",'w') as f:
    json.dump(zh_sample_triples,f,ensure_ascii=False,indent=2)
with open("data/atomic-zh/sample_test_atomic_zh_cnautomic.json",'w') as f:
    json.dump(zh_sample_triples_cnautomic,f,ensure_ascii=False,indent=2)
with open("data/atomic-zh/sample_test_atomic_zh_for_ann.tsv",'w') as f:
    for d in zh_sample_triples:
        print(f"{d['h']}\t{d['r']}\t{d['t']}\t\t",file=f)
# %%