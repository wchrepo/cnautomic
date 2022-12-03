from collections import defaultdict
import random

def default_template_arg_fn(example,name_mapping,ctx_args,arg_placeholders={"X":"<某人X>","Y":"<某人Y>","Z":"<某人Z>"}):
    ctx_args['head_item'] = name_placeholder(example['h'],**name_mapping)
    ctx_args['head_mainpart'] = name_placeholder(
        example['h'].removeprefix("<某人X>"),**name_mapping)
    if 't' in example:
        ctx_args['tail_item'] = name_placeholder(example['t'],**name_mapping)
    
    ctx_args['X'] = name_mapping.get('<某人X>')  
    ctx_args['Y'] = name_mapping.get('<某人Y>')
    ctx_args['Z'] = name_mapping.get('<某人Z>')
    return ctx_args

template_arg_fns = {
    "default":default_template_arg_fn
}


def name_placeholder(s,**kwargs):
    for k,v in kwargs.items():
        s = s.replace(k,v)
    return s

def scrub_placeholder(s,**kwargs):
    for k,v in kwargs.items():
        s = s.replace(v,k)
    return s

def example2text(example,name_mapping,example_prompt_template,arg_fn,**ctx_args):
    ctx_args = arg_fn(example,name_mapping,ctx_args)
    ctx_args = defaultdict(str,ctx_args)
    example_prompt_text = example_prompt_template.format_map(ctx_args)
    return example_prompt_text

def get_prompt(examples,generating_instance,name_mapping={},add_number=True,separator="；",
    prefix_prompt_text="",example_prompt_template="",generate_prompt_template="",
    arg_fn=default_template_arg_fn
    ):
    prompt_text = prefix_prompt_text
    for i,example in enumerate(examples):
        if isinstance(name_mapping,list):
            nm = name_mapping[i]
        else:
            nm = name_mapping
        if add_number:
            ctx = {"idx":f"{i+1}."}
        else:
            ctx = {}
        example_text = example2text(
            example,name_mapping=nm,
            example_prompt_template=example_prompt_template,arg_fn=arg_fn,
            **ctx
        )
        prompt_text+=example_text + separator
    i+=1
    if isinstance(name_mapping,list):
        nm = name_mapping[i]
    else:
        nm = name_mapping
    if add_number:
        ctx = {"idx":f"{i+1}."}
    else:
        ctx = {} 
    prompt_text+= example2text(
        generating_instance,name_mapping=nm,
        example_prompt_template=generate_prompt_template,arg_fn=arg_fn,
        **ctx
    )
    return prompt_text

def random_naming(name_placeholders=("<某人X>","<某人Y>","<某人Z>"),
    base_name_mapping={},names_pool=[],**kwargs):
    results = base_name_mapping.copy()
    chosen_names = random.sample(names_pool,len(name_placeholders))
    for placeholder,name in zip(name_placeholders,chosen_names):
        if placeholder in results:
            results[placeholder] = name
    return results