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