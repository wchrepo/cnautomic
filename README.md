# cnautomic
Resources and Codes for the EMNLP2022 paper ``CN-AutoMIC: Distilling Chinese Commonsense Knowledge from Pretrained Language Models''

## Resources

[Download link](https://mailsucaseducn-my.sharepoint.com/:f:/g/personal/wangchenhao19_mails_ucas_edu_cn/Ep4r323WZENDsWrCr7t2WF0BUZFTsZ6X6aR6gCnItNotVA?e=b2qCXa)

Content:
- CN-AutoMIC Triples
- Filter Models
- Knowledge Model

## Construction

### Environment Check

This project uses the MT5-XXL checkpoint (from huggingface) as the backbone model. In our settings, we use 3 Nvidia A6000 GPUs (48G VRAM). The model requires ~80G VRAM for inference. Please ensure your environment has a comparable total size of VRAM. 

If your GPUs have smaller VRAM size, you can modify the `config/MT5XXL_device_map.json` following the [MT5 model parallelism instruction](https://huggingface.co/docs/transformers/v4.26.0/en/model_doc/mt5#transformers.MT5Model.parallelize), and then set environment variable `DEVICE_MAP_FILE=config/MT5XXL_device_map.json` before running construction scripts.

### Construction Pipeline

Download filter_model.zip from the above link, and decompress it into the directory of this repository.


Use this command to construct the KG from scratch. 

```
bash scripts/first_build.sh CNAutoMIC
```

Use this command to extend the KG via bootstrapping iteration.

```
bash scripts/iteration_build.sh CNAutoMIC 1
```

The second argument in the command should increase 1 for each iteration.

### Customizing Construction

You can change the configuration files to modify the construction process. The config files includes:

- `config/head/default.json` The config for head generation. You can add new head categories here.
- `config/head_post/default.py` The postprocessing (normalizing) config for head generation. You can add normalization process for new head categories here.
- `config/triple/default.json` The config for triple generation. You can add new relation types here.
- `config/triple_post/default.json` The postprocessing config for triple generation. You can change the thresholds for filter models here.
- `config/iteration/default.py` The mapping config for iterative generation. You can change the mapping functions or add new ones for new relation types here.

If you create your own config files, be sure to modify the construction scripts (`scripts/heads.sh` and `scripts/triples.sh`) to use your own configs.

## Knowledge Model (CN-COMET)

Decompress the cn-automic.zip, and use huggingface transformers to load the knowledge model and infer.

```python
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
knowledge_model_path = "cn-comet/final/"

tokenizer = AutoTokenizer.from_pretrained(knowledge_model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(knowledge_model_path)

query = "某人买书 <xWant> <extra_id_0>"
results = tokenizer.batch_decode(model.generate(tokenizer(query,return_tensors='pt').input_ids.to(model.device)))
```

