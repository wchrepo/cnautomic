#%%
from collections import defaultdict
import json
import os
import re
import ipywidgets as widgets
import time
import argparse
from IPython.display import display,clear_output
from triple_utils import scrub_placeholder_plus
#%%
parser = argparse.ArgumentParser()
parser.add_argument('path',type=str)
parser.add_argument('--cookbook', default="default", type=str, 
                    help='cookbook for cooking generated triples') #You can write a customed module with `get_processing_func` function
args=parser.parse_args()


#%%
def read_instance(path,max_num=None):
    with open(os.path.join(path,"config.json")) as f:
        metadata = json.load(f)
    with open(os.path.join(path,"results.jsonl.txt")) as f:
        results = []
        for i,line in enumerate(f):
            results.append(json.loads(line))
            if max_num and i+1 == max_num:
                break
    return metadata,results
#%%

#%%

metadata,results = read_instance(args.path)
#%% 
name_mapping = metadata['base_name_mapping']
additional_names_pattern = "("+"|".join(metadata['names_pool'])+")"
#normalize and 
#remove duplicates
triple_set = set()
after_remove_duplicates = []
for result in results:
    try:
        result['triple']['t'] = scrub_placeholder_plus(result['triple']['t'],name_mapping,additional_names_pattern)
    except IndexError as e:
        #additional_placeholder_list不够用，跳过，感觉这种一般都是错误的样本
        # print(result['triple']['t'])
        continue
    triple = (result['triple']['h'],result['triple']['r'],result['triple']['t'])
    #HACK: 硬编码，这里做了名字重排处理，按说应该放到头的后处理里做
    if "<某人Z>" in triple[0]:
        if "<某人Y>" not in triple[0]:
            triple = (triple[0].replace("<某人Z>","<某人Y>"),triple[1],triple[2].replace("<某人Z>","<某人Y>"))
        elif triple[0].find("<某人Y>") > triple[0].find("<某人Z>"):
            triple = (triple[0].replace("<某人Z>","<某人TEMP>"),triple[1],triple[2].replace("<某人Z>","<某人TEMP>"))
            triple = (triple[0].replace("<某人Y>","<某人Z>"),triple[1],triple[2].replace("<某人Y>","<某人Z>"))
            triple = (triple[0].replace("<某人TEMP>","<某人Y>"),triple[1],triple[2].replace("<某人TEMP>","<某人Y>"))
        result['triple']['h'] = triple[0]
        result['triple']['t'] = triple[2]
    if triple not in triple_set:
        triple_set.add(triple)
        after_remove_duplicates.append(result)
#%% output normalized

with open(os.path.join(args.path,"normalized.jsonl.txt"),'w') as f:
    for result in after_remove_duplicates:
        print(json.dumps(result,ensure_ascii=False),file=f)