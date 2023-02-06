
#%%
import os
import json
from collections import Counter
from tqdm.auto import tqdm
import argparse
#%%合并并输出
parser = argparse.ArgumentParser()
parser.add_argument('sources',type=str,nargs='+')
parser.add_argument('--workspace',type=str,default='default')
parser.add_argument('--cookbook', default="default", type=str, 
                    help='cookbook for cooking generated triples') #You can write a customed module with `get_processing_func` function
args=parser.parse_args()
sources = args.sources
workspace = args.workspace
combined_source_data = {}
for source in sources:
    source_data = {}
    with open(os.path.join(source,'all_scored.jsonl.txt')) as f:
        for line in f:
            d = json.loads(line.rstrip())
            source_data[json.dumps(d['triple'],ensure_ascii=False)]=d
    combined_source_data.update(source_data)

#%%
out_dir = f"results/{workspace}/iteration_triple/"
if not os.path.exists(os.listdir(out_dir)):
    out_dir = out_dir + str(0)
else:
    out_dir = out_dir + str(len(os.listdir(out_dir)))
os.makedirs(out_dir,exist_ok=True)
with open(os.path.join(out_dir,'all_scored.jsonl.txt'),'w') as f:
    for triple,d in tqdm(combined_source_data.items()):
        print(json.dumps({"triple":d['triple'],"subset":d['subset'],"scores":d['scores']},ensure_ascii=False),file=f)

for split in ['high','mid','low']:
    with open(os.path.join(args.path,f"{split}_filtered.jsonl.txt"),'w') as f:
        for triple,d in tqdm(combined_source_data.items()):
            if split in d['subset']:
                print(json.dumps({"triple":d['triple'],"subset":d['subset'],"scores":d['scores']},ensure_ascii=False),file=f)
#%%
#%%
#构造一下head set
head_set = set()
for d in combined_source_data.values():
    head_set.add(d['triple']['h'])

# %%

if args.cookbook == 'default':
    def mapping_xWant(tail):
        return "<某人X>"+tail

    def mapping_xNeed(tail):
        return "<某人X>"+tail

    def mapping_xIntent(tail):
        return "<某人X>"+tail

    import jieba
    def mapping_xEffect(tail): #用了简单的词型还原，去掉“了”
        return "<某人X>"+"".join(w for w in jieba.cut(tail) if w != '了')

    def mapping_xReact(tail):
        return "<某人X>感觉" + tail

    def mapping_HinderedBy(tail):
        if tail.startswith("<某人X>"):
            return tail
        else:
            return None

    node_mapping = {
        "xAttr": None,
        "xWant": (mapping_xWant,"voluntary_occurences"),
        "xNeed": (mapping_xNeed,"voluntary_occurences"),
        "xIntent": (mapping_xIntent,"voluntary_occurences"),
        "xEffect": (mapping_xEffect,"involuntary_occurences"),
        "xReact": (mapping_xReact,"states"),
        "HinderedBy": (mapping_HinderedBy,"involuntary_occurences"),
    }
    ensure_occurence_more_than_n = 3
else:
    import importlib
    cookbook = importlib.import_module(args.cookbook)
    node_mapping=cookbook.node_mapping
    ensure_occurence_more_than_n = cookbook.ensure_occurence_more_than_n
#%% count tail for each rel
analysis_rel_tail = {}
for d in combined_source_data.values():
    if 'mid' in d['subset']:
        if d['triple']['r'] not in analysis_rel_tail:
            analysis_rel_tail[d['triple']['r']] = Counter()
        analysis_rel_tail[d['triple']['r']][d['triple']['t']] += 1


# # %% 

analysis_rel_tail_sorted = {}
for rel,rel_stat in analysis_rel_tail.items():
    total = sum(v for v in rel_stat.values())
    analysis_rel_tail_sorted[rel] = []
    for d in sorted(((k,v,v/total) for k,v in rel_stat.items()),key=lambda x:x[1],reverse=True):
        if node_mapping.get(rel) is None:
            analysis_rel_tail_sorted[rel].append(d + (None,False))
        else:
            m_func,m_category = node_mapping[rel]
            mapped = m_func(d[0])
            analysis_rel_tail_sorted[rel].append(d+(mapped,mapped in head_set))
# %%


def get_tail_occurence_more_than_n(rel_tail_sorted,n):
    to_return = []
    for d in rel_tail_sorted:
        if d[1] < n:
            break
        to_return.append(d)
    return to_return
#%%

highfreq_tails = {}
for rel,rel_stat in analysis_rel_tail_sorted.items():
    highfreq_tails[rel] = get_tail_occurence_more_than_n(rel_stat,ensure_occurence_more_than_n)
for rel,data in highfreq_tails.items():
    print(rel,len(data))
print(sum(len(data) for data in highfreq_tails.values()))


#%%
# mapping_records = {} #rel,tail -> head eventuality 
mapping_results = {
    "voluntary_occurences":[],
    "involuntary_occurences":[],
    "states":[],
}
# top_in_all_mapping_stat = {}
for rel,data in highfreq_tails.items():
    if node_mapping.get(rel) is None:
        continue
    # top_in_all_mapping_stat[rel] = []
    for d in data:
        m_func,m_category = node_mapping[rel]
        mapped = m_func(d[0])
        if mapped:
            # mapping_records[(rel,d[0])] = mapped
            mapping_results[m_category].append(mapped)
        # top_in_all_mapping_stat[rel].append(d+(mapped,mapped in head_set))

print("After mapping:")
for k,v in mapping_results.items():
    print(k,len(v))

print("After remove duplicates")
mapping_results_deduplicates = {}
for k,v in mapping_results.items():
    mapping_results_deduplicates[k] = list(set(v)-head_set)
    print(len(mapping_results_deduplicates[k]))
#%%
#Output
out_dir = f"results/{workspace}/iteration_heads/"
if not os.path.exists(os.listdir(out_dir)):
    out_dir = out_dir + str(1)
else:
    out_dir = out_dir + str(1+len(os.listdir(out_dir)))
os.makedirs(out_dir,exist_ok=True)
for category,results in mapping_results_deduplicates.items():
    with open(os.path.join(out_dir,f"{category}.jsonl.txt"),'w') as f:
        print(json.dumps({"sources":sources,"keep_n":ensure_occurence_more_than_n}),file=f)
        for r in results:
            print(r,file=f)
# %%
