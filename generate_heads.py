#%% Load library
from model_wrappers import CPM2,MT5XXL
from head_item_examples import *
import iwexp
import random
import numpy as np
import json
from tqdm.auto import tqdm
import mlflow
import time
import argparse
import ast
import os
#%%
parser = argparse.ArgumentParser()
parser.add_argument('category',type=str,help="category of generated head") 
parser.add_argument('--config', default=None, type=str,nargs='*',
    help='config for generate inference')
parser.add_argument('--workspace',default="default",type=str,
    help="an optional workspace (sub-directory) name. The generated results will be saved at `results/{workspace}/heads/{category}/`")
parser.add_argument('--output_filename',default=None,type=str,help="output file name. If omitted, the output file name will be set according to current time.")
parser.add_argument('--force_params',default=None,type=str,help="changing hyperparameters using python dict syntax. Example: {'random_seed':2}")
parser.add_argument("--note",type=str,default="", help="add a note about the experiment.")
args=parser.parse_args()
if args.force_params:
    force_params = ast.literal_eval(args.force_params)
else:
    force_params = {}
workspace = args.workspace
#%% config
cfg = iwexp.TracingConfig(globals(),load_path=args.config,force_update_dict=force_params)
with cfg:
    random_seed = 1
    n_example_in = 10
    n_generation = 100
    n_iteration = 2000 
    category = args.category #"voluntary_occurences","involuntary_occurences","states"
    name_mapping = {
        "<某人X>":"张三",
        "<某人Y>":"李四",
        "<某人Z>":"王五",
        "<某样东西Y>":"某样东西",
    }
    separator = "\n"
    add_number= True # add order numbers before examples
    prefix_prompt_text = f'' #prefix prompt is before all examples
    example_prompt_text = "" #example prompt is before each example (but after order numbers) 
    generate_prompt_text = name_mapping["<某人X>"] + " <extra_id_0>" # generate prompt is at the end of prompt text.
    #generate params
    generate_params = {
        "top_p":0.9,
        # "top_n":1,
        "max_tokens":16,
        "temperature":1.0,
        "stop_tokens":['；','。',"（","\n",";","…"," "], # If these tokens appear in the result, it will be truncated with these tokens and regarded as `stop=True` and .
    }
#%% 
# utils
def example2text(example,example_prompt_text="",incomplete=False,number=None,separator="；"):
    text = ""
    if number:
        text += f"{number}. "
    text += f"{example_prompt_text}{example}"    
    if not incomplete:
        text += separator
    return text
    
def name_placeholder(s,**kwargs):
    for k,v in kwargs.items():
        s = s.replace(k,v)
    return s

def scrub_placeholder(s,**kwargs):
    for k,v in kwargs.items():
        s = s.replace(v,k)
    return s

def get_prompt(examples,add_number=True,name_mapping={},separator="；",
    prefix_prompt_text="",example_prompt_text="",generate_prompt_text=""):
    prompt_text = prefix_prompt_text
    for i,example in enumerate(examples):
        if isinstance(name_mapping,list):
            nm = name_mapping[i]
        else:
            nm = name_mapping
        example = name_placeholder(example,**nm)
        if add_number:
            number = i+1
        else:
            number = None
        example_text = example2text(
            example,example_prompt_text=example_prompt_text,
            incomplete=False,number=number,separator=separator
            )
        prompt_text+=example_text
    i+=1
    if add_number:
        number = i+1
    else:
        number = None    
    prompt_text+= example2text(
        generate_prompt_text,example_prompt_text=example_prompt_text,
        incomplete=True,number=number,separator=separator
    )
    return prompt_text
#%% prepare model
model = MT5XXL()
model.prepare()
#%%
# Generate 
random.seed(random_seed)
np.random.seed(random_seed)

if category == "voluntary_occurences":
    all_seeds = voluntary_occurences
elif category == "involuntary_occurences":
    all_seeds = involuntary_occurences
elif category == "states":
    all_seeds = states
else:
    raise NotImplementedError(f"The seeds of {category} are not defined")


prompt_params = {
    "name_mapping":name_mapping,
    "separator":separator,
    "add_number":add_number,
    "prefix_prompt_text":prefix_prompt_text,
    "example_prompt_text":example_prompt_text,
    "generate_prompt_text":generate_prompt_text
}

run_params = {
    "random_seed":random_seed,
    "n_example_in":n_example_in,
    "n_generation":n_generation,
    "n_iteration":n_iteration
}


def voluntary_prompt(examples):
    return get_prompt(examples,**prompt_params)

#%% generate
mlflow.set_experiment("MT5_head_gen_v1")
mlflow.start_run()
mlflow.set_tags({
    "model":"MT5XXL",
    "target":"head",
    "category":f"{category}",
    "note":""
})
mlflow.log_params({f"prompt_params.{k}":v for k,v in prompt_params.items()})
mlflow.log_params({f"run_params.{k}":v for k,v in run_params.items()})
mlflow.log_params(generate_params)
os.makedirs(f"results/{workspace}/heads/{category}/",exist_ok=True)
output_filename = args.output_filename
if output_filename is None:
    output_filename = time.strftime('%Y%m%d_%a_%H:%M:%S')
output_path = f"results/{workspace}/heads/{category}/{output_filename}.jsonl.txt"
with open(output_path,'w') as f:
    print(json.dumps(
            {"prompt_params":prompt_params,
            "run_params":run_params,
            "generate_params":generate_params},ensure_ascii=False
        ),
        file=f)
    for i in tqdm(range(n_iteration),desc="iter"): 
        examples = random.sample(all_seeds,k=n_example_in)
        input_prompt = voluntary_prompt(examples)
        results = model.n_generate(input_prompt,n_generation=n_generation,**generate_params)
        for result in results:
            if result['stop']:
                print(json.dumps(result,ensure_ascii=False),file=f)
mlflow.log_artifact(output_path)
mlflow.end_run()
