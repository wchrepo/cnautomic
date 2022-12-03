"""
I wanna experiment: An experiment configuration tool for lazy me.
Its main use is to save the global variables defined in a codespan.
Because I am too lazy, the module only implements basic features,
such as tracing `globals()`, loading config files (.json), etc.

--By WCH
"""

import json
import os
import logging
class TracingConfig(dict):
    def __init__(self,tracing_obj,load_path=None,force_update_dict={},lazy_update=False,check_load_cfg="no",check_force_cfg="error") -> None:
        self.cfg_vars = set()
        self.tracing_obj = tracing_obj
        self.load_path = load_path
        if self.load_path:
            if isinstance(self.load_path,str):
                self.load_path = [self.load_path]
            self.load_cfgs = []
            for path in self.load_path:
                assert os.path.exists(path),f"{path} does not exist"
                with open(path) as f:
                    if path.endswith(".json"):
                        self.load_cfgs.append((path,json.load(f)))
        self.force_update_dict = force_update_dict
        self.lazy_update = lazy_update
        self.check_load_cfg = check_load_cfg
        self.check_force_cfg = check_force_cfg
    
    def __enter__(self):
        self._old_vars = set(k for k in self.tracing_obj if not k.startswith('_'))

    def __exit__(self,exc_type, exc_val, exc_tb):
        self._new_vars = set(k for k in self.tracing_obj if not k.startswith('_'))
        self.cfg_vars.update(self._new_vars-self._old_vars)
        self.update({k:v for k,v in self.tracing_obj.items() if k in self.cfg_vars})
        if not self.lazy_update:
            self.update_from_load_cfg()
            self.update_from_force_cfg()
        self.tracing_obj.update(self)
        return False
    

    def _update_from_cfg(self,cfg,cfg_name="",check="warning"):
        def not_found_key(k):
            if check=='no':
                pass
            elif check == 'warning':
                logging.warning(f"{cfg_name}: key `{k}` is not in the current config")
            else:
                assert k in self,f"{cfg_name}: key `{k}` is not in the current config"
        for k,v in cfg.items():
            if '.' not in k:
                if k not in self:
                    not_found_key(k,check)
                self[k]=v
            else:
                keys = k.split('.')
                nested = self
                is_not_found = False
                for key in keys[:-1]:
                    if key not in nested:
                        if not is_not_found:
                            not_found_key(k)
                            is_not_found = True
                        nested[key] = {}
                    nested = nested[key]
                if keys[-1] not in nested and not is_not_found:
                    not_found_key(k)
                nested[keys[-1]]=v


    def update_from_load_cfg(self):
        if self.load_path:
            for cfg_name,cfg in self.load_cfgs:
                self._update_from_cfg(cfg,cfg_name=cfg_name,check=self.check_load_cfg)

    def update_from_force_cfg(self):
        #force cfg
        self._update_from_cfg(self.force_update_dict,cfg_name="cli params:",check=self.check_force_cfg)

    def __call__(self,group="",tracing_obj=None):
        if tracing_obj:
            self.tracing_obj = tracing_obj
        if group: #TODO: Grouping config. NOT IMPLEMENTED YET 
            self.group = group
        return self
    
    def dump(self,dump_path):
        if not os.path.exists(os.path.dirname(dump_path)):
            os.makedirs(os.path.dirname(dump_path))
        with open(dump_path,'w') as f:
            json.dump(self,f,ensure_ascii=False,indent=2)
