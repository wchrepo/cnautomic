#%% Load library
from model_wrappers import CPM2, MT5XXL
from head_item_examples import *
from triple_examples import default_examples
from triple_utils import *
from collections import defaultdict
import random
import functools
import numpy as np
import json
from tqdm.auto import tqdm
import mlflow
import time
import iwexp
import argparse
import ast
import os

#%%
parser = argparse.ArgumentParser()
parser.add_argument('source',type=str,default="voluntary_occurences",nargs="?",help="Path of generated heads.") #要么是head_item_examples里定义的例子，要么是指定路径
parser.add_argument('workspace',type=str,default="default",
    help="an optional workspace (sub-directory) name. The generated results will be saved at `results/{workspace}/triple/{category}/`")
parser.add_argument('--config', default=None, type=str,nargs='*',
                    help='config files for generate inference')
parser.add_argument('--force_params',default=None,type=str)
parser.add_argument("--intent",type=str,default="")
args=parser.parse_args()
if args.force_params:
    force_params = ast.literal_eval(args.force_params)
else:
    force_params = {}
workspace = args.workspace
#%%
cfg = iwexp.TracingConfig(globals(),load_path=args.config,force_update_dict=force_params)
with cfg:
    #共享的运行参数
    source = args.source
    model_name = "MT5"
    exp_category = "iteration" #"voluntary_occurences"
    exp_relation = ["xWant","xAttr","xIntent","xEffect","xReact","xNeed","HinderedBy"]
    random_seed = 1
    names_pool = [
        "张伟",
        "李强",
        "王勇",
        "刘洋",
        "王磊",
        "李军",
        "刘杰",
        "王刚",
        "张鹏",
        "李辉",
        "王芳",
        "李娜",
        "李秀英",
        "张静",
        "张敏",
        "李娟",
        "李霞",
        "张燕",
        "王桂英",
        "刘娟",
    ]
    base_name_mapping = {
        "<某人X>":"张三",
        "<某人Y>":"李四",
        "<某人Z>":"王五",
        "<某样东西Y>":"某样东西",
    }
    placeholders_for_random_naming = ["<某人X>","<某人Y>","<某人Z>"]
    template_arg_fn = "default"
    template_arg_fn_setting = {
        "arg_placeholders":{"X":"<某人X>","Y":"<某人Y>","Z":"<某人Z>"}
    }
    use_random_naming = True
    use_random_examples = False
    n_example_in = 8
    if model_name == "MT5":
        generating_placeholder = " <extra_id_0>；"
    else:
        generating_placeholder = ""
    #生成参数
    generate_params = {
        "top_p":0.7,
        # "top_n":1,
        "max_tokens":16,
        "temperature":1.0,
        # "frequency_penalty":0.0,
        # "presence_penalty":0.0,
        "stop_tokens":['；','。',"（","\n",";","…",",","."],
        # "stop_tokens":['；','。','<unk>'], #去掉了括号，后处理再去
        "n_generation":10, #从此作为generate参数，方便以后并行
    }
    relation_params = {
        "xWant":{
            # "pre_prompt":{},
            # "examples":[],
            "prompt":{
                "add_number":True,
                "separator":"；",
                "prefix_prompt_text":"请填写在以下事件后人物想做的事，例如：",
                "example_prompt_template":"{idx} {head_item}，在此之后，{X}想要{tail_item}",
                "generate_prompt_template":"{idx} {head_item}，在此之后，{X}想要{tail_item}",
            }
        },
        "xAttr":{
            # "pre_prompt":{},
            # "examples":[],
            "prompt":{
                "add_number":True,
                "separator":"；",
                "prefix_prompt_text":"请写出以下事件中，可以看出的人物特点。例如：",
                "example_prompt_template":"{idx} {head_item}，据此，可以看出{X}是{tail_item}",
                "generate_prompt_template":"{idx} {head_item}，据此，可以看出{X}是{tail_item}",
            }
        },
        "xEffect":{
            # "pre_prompt":{},
            # "examples":[],
            "prompt":{
                "add_number":True,
                "separator":"；",
                "prefix_prompt_text":"请写出以下事件对人物的影响，例如：",
                "example_prompt_template":"{idx} {head_item}。结果，{X}{tail_item}",
                "generate_prompt_template":"{idx} {head_item}。结果，{X}{tail_item}",
            }
        },
        "xReact":{
            # "pre_prompt":{},
            # "examples":[],
            "prompt":{
                "add_number":True,
                "separator":"；",
                "prefix_prompt_text":"请写出人物在以下事件后的反应、感受，例如：",
                "example_prompt_template":"{idx} {head_item}，对此，{X}感觉{tail_item}",
                "generate_prompt_template":"{idx} {head_item}，对此，{X}感觉{tail_item}",
            }
        },
        "xNeed":{
            # "pre_prompt":{},
            # "examples":[],
            "prompt":{
                "add_number":True,
                "separator":"；",
                "prefix_prompt_text":"请填写事件发生的前提需求，例如：",
                "example_prompt_template":"{idx} 在{head_item}之前，{X}需要{tail_item}",
                "generate_prompt_template":"{idx} 在{head_item}之前，{X}需要{tail_item}",
            }
        },
        "xIntent":{
            # "pre_prompt":{},
            # "examples":[],
            "prompt":{
                "add_number":True,
                "separator":"；",
                "prefix_prompt_text":"请填写人物的意图，例如：",
                "example_prompt_template":"{idx} {head_item}，{X}的意图是{tail_item}",
                "generate_prompt_template":"{idx} {head_item}，{X}的意图是{tail_item}",
            }
        },
        "HinderedBy":{
            # "pre_prompt":{},
            # "examples":[],
            "prompt":{
                "add_number":True,
                "separator":"；",
                "prefix_prompt_text":"什么情况会阻碍如下事件的发生？例如：",
                "example_prompt_template":"{idx} {head_item}，这受到阻碍，因为{tail_item}",
                "generate_prompt_template":"{idx} {head_item}，这受到阻碍，因为{tail_item}",
                # "example_prompt_template": "{idx} {head_item}，但受到阻碍没有顺利进行，因为{tail_item}",
                # "generate_prompt_template": "{idx} {head_item}，但受到阻碍没有顺利进行，因为{tail_item}",
            }
        },
    }
#%%
#

#%%
run_params = {
    "random_seed":random_seed,
}
pre_prompt_params = { #The settings of preparation of examples. They can be overrided by relation-level setting.
    "names_pool":names_pool,
    "base_name_mapping":base_name_mapping,
    "use_random_naming":use_random_naming,
    "use_random_examples":use_random_examples,
    "n_example_in":n_example_in,
    "generating_placeholder":generating_placeholder,
}


#%%
arg_fn = functools.partial(template_arg_fns['template_arg_fn'],**template_arg_fn_setting)
def prompt_fn(head_item,relation):
    #使用relation_params里的参数覆盖外面的参数，比如n_example_in
    override_params = relation_params[relation].get("pre_prompt",{})
    locals().update(override_params) 
    examples = relation_params[relation].get("examples",default_examples[relation])
    #随机命名
    if use_random_naming:
        name_mapping = [
            random_naming(
                name_placeholders=placeholders_for_random_naming,
                base_name_mapping=base_name_mapping,
                names_pool=names_pool
            ) 
            for i in range(n_example_in)
        ] + [base_name_mapping]
    else:
        name_mapping = base_name_mapping
    #随机例子
    if use_random_examples:
        examples = random.sample(examples,n_example_in)
    else:
        examples = examples[:n_example_in]
    return get_prompt(examples,{'h':head_item,'r':relation,"t":generating_placeholder},
        name_mapping=name_mapping,
        arg_fn = arg_fn,
        **relation_params[relation]['prompt']
    )
#%% load model
if model_name == "CPM2":
    model = CPM2()
    model.prepare()
elif model_name == "MT5":
    model = MT5XXL()
    model.prepare()
else:
    raise NotImplementedError(model_name)
#%% load source data
source_mapping = {
    "voluntary_occurences":voluntary_occurences,
    "states":states,
    "involuntary_occurences":involuntary_occurences,
}
if source in source_mapping:
    source_data = source_mapping[source]
else:
    with open(source) as f:
        f.readline()
        source_data = []
        for line in f:
            source_data.append(line.strip())
#%%
output_path = f"results/{workspace}/triples/{exp_category}/{time.strftime('%Y%m%d_%a_%H:%M:%S')}/"
mlflow.set_experiment("MT5_triple_gen")
mlflow.start_run()
mlflow.set_tags({
    "source":source,
    "source_length":len(source_data),
    "model":model_name,
    "target":"triple",
    "category":exp_category,
    "relation":exp_relation,
    "intent":args.intent,
    "output_path":output_path,
})
mlflow.log_params(iwexp.flatten_dict(run_params,"run_params",1))
mlflow.log_params(iwexp.flatten_dict(pre_prompt_params,"pre_prompt_params",1))
mlflow.log_params(iwexp.flatten_dict(relation_params,"relation_params",3))
mlflow.log_params(generate_params)
#%%
os.makedirs(output_path)
cfg.dump(os.path.join(output_path,"config.json"))
with open(os.path.join(output_path,"results.jsonl.txt"),'w') as f:
    for head_item in tqdm(source_data,desc="head instance"): #或者也可以考虑像ATOMIC10X一样，先shuffle再按一个epoch进行
        for relation in exp_relation:
            #TODO 之后做个更全面的允许规则系统，而不是这里硬编码
            if relation.startswith("y") and "<某人Y>" not in head_item:
                continue
            input_prompt = prompt_fn(head_item,relation)
            if args.intent == 'debug':
                tqdm.write(input_prompt)
            generation_results = model.n_generate(input_prompt,**generate_params)
            if args.intent == "debug":
                tqdm.write(str([result['text'] for result in generation_results]))
            for generation_result in generation_results:
                if generation_result['stop']:
                    result = {
                        "triple":{
                            "h":head_item,
                            "r":relation,
                            "t":generation_result['text'],
                        },
                        "gen":generation_result
                    }
                    print(json.dumps(result,ensure_ascii=False),file=f)
#%%
mlflow.log_artifact(os.path.join(output_path,"results.jsonl.txt"))
mlflow.end_run()