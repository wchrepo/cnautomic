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