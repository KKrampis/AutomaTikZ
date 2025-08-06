from os.path import isfile, join

from datasets import DownloadManager as DL
from peft import PeftModel, LoraConfig # type: ignore
import torch
from transformers import AutoConfig
from transformers.utils.hub import is_remote_url
from huggingface_hub import hf_hub_download

from .. import train
from ..model.clima import register
from ..util import merge_and_unload, temporary_change_attributes

def load(path, **kwargs):
    # Support Hugging Face Inference Endpoint (shortcut); accept custom scheme or full API URL
    if isinstance(path, str):
        # custom scheme: hf-inference://repo_id
        if path.startswith("hf-inference://"):
            from huggingface_hub import InferenceApi

            repo_id = path.split("hf-inference://", 1)[1]
            token = kwargs.pop("token", None)
            return InferenceApi(repo_id=repo_id, token=token), None
        # full API URL: https://api-inference.huggingface.co/models/repo_id
        prefix = "https://api-inference.huggingface.co/models/"
        if path.startswith(prefix):
            from huggingface_hub import InferenceApi

            repo_id = path.removeprefix(prefix)
            token = kwargs.pop("token", None)
            return InferenceApi(repo_id=repo_id, token=token), None

    register()
    if (is_remote:=is_remote_url(path)) or isfile(path): # treat it as a pretrained mm projector
        hidden_size = len(torch.load(DL().download(path) if is_remote else path)['model.mm_projector.weight']) # type: ignore
        size_dict = {
            4096: "7b",
            5120: "13b",
            6656: "30b",
            8192: "65b"
        }
        model, tokenizer = train.clima.load(pretrain_mm_mlp_adapter=path, size=size_dict[hidden_size], model_kwargs=kwargs)
    else:
        # try adapter_config.json in local folder or on Hugging Face Hub
        conf_file = None
        local_conf = join(path, "adapter_config.json")
        if isfile(local_conf):
            conf_file = local_conf
        else:
            try:
                conf_file = hf_hub_download(repo_id=path, filename="adapter_config.json")
            except Exception:
                conf_file = None
        if conf_file:
            conf = LoraConfig.from_json_file(conf_file)
            base_model = conf["base_model_name_or_path"]
            model_type = conf.pop("model_type", AutoConfig.from_pretrained(base_model).model_type)
            model, tokenizer = getattr(train, model_type).load(base_model=base_model, model_kwargs=kwargs)
            # load adapter into RAM before merging to save GPU memory
            with temporary_change_attributes(torch.cuda, is_available=lambda: False):
                model = merge_and_unload(PeftModel.from_pretrained(
                    model,
                    path,
                    torch_dtype=model.config.torch_dtype,
                    config=LoraConfig(**conf),
                    **kwargs
                ))
        else:
            raise ValueError(f"Cannot load model from {path}.")

    return model.eval(), tokenizer # type: ignore
