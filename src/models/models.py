import torch
import torch.distributed as dist
import torch.nn as nn

from src.models.gpt import (LMModel, load_openai_pretrained_model)


def make_model(opt, n_vocab, n_ctx, n_special, load=True,
               return_acts=True, return_probs=False,
               clf_token="<CLASS>", answer_size=None, compile_model=False):
    print(n_ctx)
    if opt.exp == "generation":
        model = LMModel(
            opt.net, n_vocab, n_ctx, return_acts=return_acts,
            return_probs=return_probs)
    elif opt.exp == "classification":
        model = ClfModel(
            opt.net, n_vocab, n_ctx, clf_token, answer_size)
    if load:
        print("LOADING PRETRAINED TRANSFORMER")
        load_openai_pretrained_model(
            model.transformer, n_ctx=n_ctx, n_special=n_special)
    if compile_model:
        model = torch.compile(model)
    return model



def multi_gpu(model, devices):
    # devices deve ser uma lista de índices de GPU
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = devices[0] if isinstance(devices, (list, tuple)) else devices
    torch.cuda.set_device(local_rank)
    model = model.to(local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    return model


def load_state_dict(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        new_state_dict = {i[len("module."):]: j for i, j in state_dict.items()}
        model.load_state_dict(new_state_dict)
