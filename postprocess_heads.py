#%%
import json
import os
import argparse

#%%
parser = argparse.ArgumentParser()
parser.add_argument('path',type=str)
parser.add_argument('--cookbook', default="default", type=str, 
                    help='cookbook for cooking generated heads') #You can write a customed module with `get_processing_func` function
parser.add_argument("--category",default=None, type=str,help="category of heads. If omitted, the category will be detected according to path.")
parser.add_argument("--score_ratio",type=float,default=0.7,help="keeping ratio according to the generation scores (nll)")
args=parser.parse_args()

#%%
category = args.category
if category is None:
    category = args.path.split('/')[-2]
input_file_path = args.path
output_file_path = os.path.join(os.path.dirname(input_file_path),f'processed_{os.path.basename(input_file_path)}')
ratio = args.score_ratio

#%%
def read_instances(path):
    instances = []
    with open(path) as f:
        metadata = json.loads(f.readline())
        for line in f:
            instance = json.loads(line)
            instances.append(instance)
    count_instance = len(instances)
    result = {
        "metadata":metadata,
        "result":instances,
    }
    print(f"Read {count_instance} from {path}")
    return result


#%%
result = read_instances(args.path)

if args.cookbook == 'default':
    name_mapping = result['metadata']['prompt_params']['name_mapping']
    def scrub_placeholder(s,**kwargs):
        for k,v in kwargs.items():
            s = s.replace(v,k)
        return s
    def get_processing_func(category):
        if category == 'voluntary_occurences':
            # remove_rules tell when to remove a generated results. They are applied sequentially. 
            # If any remove_rule returns True, the result will be removed.
            remove_rules = [
                lambda x:x['text'].startswith(("的","被","、","受到")), # forbidden prefix
                lambda x: any(not( # allowed characters
                    c.isalnum() or
                    '\u4e00' <= c <= '\u9fa5' or
                    c in "‘’“”："
                    ) 
                    for c in x['text']),
                lambda x: len(x['text']) == 0 # not blank string
            ]
            # normalize the generated results. For example, remove specific names with placeholders.
            def normalize(instances):
                normalized = []
                for instance in instances:
                    instance = scrub_placeholder(instance,**name_mapping)
                    instance = f"<某人X>{instance}"
                    normalized.append(instance)
                return normalized
        elif category == 'states':
            remove_rules = [
                lambda x:x['text'].startswith(("、",)),
                lambda x: any(not(
                    c.isalnum() or
                    '\u4e00' <= c <= '\u9fa5' or
                    c in "‘’“”：,"
                    ) 
                    for c in x['text']),
                lambda x: len(x['text']) == 0
            ]
            def normalize(instances):
                normalized = []
                for instance in instances:
                    instance = scrub_placeholder(instance,**name_mapping)
                    instance = f"<某人X>{instance}"
                    normalized.append(instance)
                return normalized
        elif category == 'involuntary_occurences':
            remove_rules = [
                lambda x:x['text'].startswith(("、",)),
                lambda x: any(not(
                    c.isalnum() or
                    '\u4e00' <= c <= '\u9fa5' or
                    c in "‘’“”：,"
                    ) 
                    for c in x['text']),
                lambda x: len(x['text']) == 0
            ]
            def normalize(instances):
                normalized = []
                for instance in instances:
                    instance = scrub_placeholder(instance,**name_mapping)
                    instance = f"<某人X>{instance}"
                    normalized.append(instance)
                return normalized
        else:
            raise NotImplementedError
        return remove_rules,normalize
else:
    import importlib
    cookbook = importlib.import_module(args.cookbook)
    remove_rules, normalize = cookbook.get_processing_func(category,result['metadata'])



#%% filter and normalize
print(f"===={input_file_path}====")
result['result'].sort(key=lambda x:x['nll'])
instances = result['result']
print(f"instances: {len(instances)}")
filtered_instances = instances[:int(len(instances)*ratio)]
print(f"after ratio filter: {len(filtered_instances)}")
filtered_instances = [
    instance for instance in filtered_instances
    if all(not rule(instance) for rule in remove_rules)
]
print(f"after rule filter: {len(filtered_instances)}")
after_removing_duplicates = list(set(instance['text'] for instance in filtered_instances))
print(f"after remove duplicates: {len(after_removing_duplicates)}")
print(f"final proportion: {len(after_removing_duplicates)/len(instances):.2f}")
normalized = normalize(after_removing_duplicates)

metadata = result['metadata']
metadata.update({"postprocess":{
    "category":category,
    "ratio":ratio,
    "input_file_path":input_file_path,
    "output_file_path":output_file_path
}
})
if not os.path.exists(os.path.dirname(output_file_path)):
    os.makedirs(os.path.dirname(output_file_path))
with open(output_file_path,'w') as f:
    print(json.dumps(metadata,ensure_ascii=False),file=f)
    for line in normalized:
        print(line,file=f)