
import re
import torch
import os
class UnderlyingModel:

    def __init__(self) -> None:
        self.is_prepared = False

    def prepare(self):
        raise NotImplementedError
    def generate(self,inputs,**kwargs):
        raise NotImplementedError
    def n_generate(self,inputs,n_generation=1,**kwargs):
        result = []
        for i in range(n_generation):
            result.append(self.generate(inputs,**kwargs))
        return result

class CPM2(UnderlyingModel):
    def __init__(self) -> None:
        super().__init__()
    
    def prepare(self):
        import bminf
        self.cpm2 = bminf.models.CPM2()
        self.is_prepared = True

    def generate(self, inputs, stop_tokens=[],**kwargs):
        stop_tokens = ["<unk>"] + stop_tokens
        result,stop,nll =  self.cpm2.generate(inputs,**kwargs)
        return {
            "text":result,
            "stop":stop,
            "nll":nll,
        }

_HF_kwargs_mapping = {
    "n_generation":"num_return_sequences",
    "max_tokens":"max_length",
}

class HFS2SModel(UnderlyingModel):
    def __init__(self,pretrained_model_name_or_path) -> None:
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
    def prepare(self,**kwargs):
        import transformers
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.pretrained_model_name_or_path
        )
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            self.pretrained_model_name_or_path,**kwargs
        )
        # if parallel:
        #     parallelize(self.model,**parallelformers_kwargs)
    
    def n_generate(self,inputs,n_generation=1,end_token=None,max_tokens=None,
        stop_tokens=[],skip_special_tokens=True,return_score="nll",**kwargs):
        input_ids = self.tokenizer(inputs,return_tensors='pt').input_ids.to(self.model.device)
        end_token = end_token or self.tokenizer.eos_token
        eos_token_id = self.tokenizer(end_token).input_ids[0] if end_token else None
        num_return_sequences = n_generation
        max_length = max_tokens
        output = self.model.generate(input_ids,eos_token_id = eos_token_id,
            num_return_sequences = num_return_sequences,
            max_length = max_length,
            do_sample=True,
            **kwargs
        )
        has_stopped = ((output[:,-1]==self.tokenizer.eos_token_id) | 
            (output[:,-1]==self.tokenizer.pad_token_id)).tolist()
        results = self.tokenizer.batch_decode(output,skip_special_tokens=skip_special_tokens)
        #TODO: Maybe I should check whether the generation has stop before tokenizer decoding? 
        stop_tokens_pattern = "|".join(re.escape(stop_token) for stop_token in stop_tokens)
        pruned_results = []
        result_dicts = []
        for i,result in enumerate(results):
            if stop_tokens_pattern:
                match = re.search(stop_tokens_pattern,result)
                if match:
                    result = result[:match.start()].strip()
                    has_stopped[i]=True
            pruned_results.append(result)
            result_dicts.append({"text":result,"stop":has_stopped[i]})
        if return_score == 'nll':
            with torch.no_grad():
                labels = self.tokenizer(pruned_results,return_tensors='pt',
                    padding=True).input_ids.to(device=self.model.device)
                labels[(labels == self.tokenizer.eos_token_id)| 
                    (labels == self.tokenizer.pad_token_id)]=-100
                check_output = self.model(input_ids.expand(len(labels),-1),
                    labels=labels)
                nll = torch.nn.functional.cross_entropy(
                    check_output.logits.transpose(1,2),labels,reduction='none'
                ).sum(dim=-1).tolist()
            for i,d in enumerate(result_dicts):
                d['nll'] = nll[i]
       
        return result_dicts
    def generate(self,inputs,end_token=None,**kwargs):
        return self.n_generate(inputs,n_generation=1,end_token=end_token,**kwargs)[0]

class MT5XXL(HFS2SModel):
    def __init__(self) -> None:
        super().__init__("google/mt5-xxl")
    def prepare(self,device_map=None):
        super().prepare()
        #WARNING: Hardcode. device_map could be in a config file.
        # device_map = {
        #     0: [0, 1, 2],
        #     1: [3, 4, 5, 6, 7, 8, 9, 10, 11],
        #     2: [12, 13, 14, 15, 16, 17,18,19,20,21,22,23]
        # }
        if "DEVICE_MAP_FILE" in os.environ:
            with open(os.environ['DEVICE_MAP_FILE']) as f: 
                device_map = json.load(f)
                device_map = {(int(k) if k.isdecimal() else k):v for k,v in device_map.items()}items()}
        self.model.parallelize(device_map)
    def n_generate(self, inputs, n_generation=1, end_token=None, max_tokens=None, stop_tokens=[], **kwargs):
        #
        results = super().n_generate(inputs, n_generation, end_token, max_tokens,
            stop_tokens=stop_tokens+["<extra_id_1>"], **kwargs)
        for result in results:
            result['text'] = result['text'].removeprefix("<extra_id_0>")
        return results
        
