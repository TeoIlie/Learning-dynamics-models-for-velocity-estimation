import json
import os
import pickle
from pathlib import Path
import wandb
import torch
from utils.argparser import get_parser
from utils.res_model_training import create_residual_model
from utils.base_model_training import create_base_model
from utils.create_noise_model import get_noise_model


def get_wandb_model(model_name):
    model_path = Path('artifacts') / model_name

    if not model_path.exists():
        run = wandb.init()
        artifact = run.use_artifact(
            f'f1tenth/ukf_all_in_one/{model_name}', type='model')
        artifact_dir = artifact.download()
        run.finish()

    config = json.load(open(model_path / "args.json", "r"))
    args = get_parser()
    args = args.parse_args([])
    args.__dict__.update(config)

    device = torch.device(args.common_device)

    if args.res_enable:
        model, _ = create_residual_model(args)
    else:
        model, _ = create_base_model(args)

    noise_model = get_noise_model(args)
    model_list = os.listdir(model_path)
    model_list.sort()

    print(f"model path: {model_list}")
    noise_model_path = model_path / model_list[-1]
    model_path = model_path / model_list[-2]

    model_dict = torch.load(model_path)
    noise_model_dict = torch.load(noise_model_path)

    model.load_state_dict(model_dict, strict=False)
    noise_model.load_state_dict(noise_model_dict)

    # raise NotImplementedError("This function is not implemented yet")

    return args, model, noise_model
