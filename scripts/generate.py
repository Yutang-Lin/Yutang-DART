import os
import sys
import json

# Add current directory to system path
DART_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(DART_PATH)

from mld.rollout_mld_detailed import (
    rollout,
    RolloutArgs,
    load_mld,
    create_gaussian_diffusion,
    SinglePrimitiveDataset,
    Path
)
import tyro
import torch
import random
import numpy as np
from dataclasses import asdict
                
def main():
    rollout_args = tyro.cli(RolloutArgs)
    if rollout_args.save_dir is None:
        rollout_args.save_dir = 'output/motion_generation'
    rollout_args.respacing = ''
    rollout_args.guidance_param = 5.0
    rollout_args.export_smpl = 1
    rollout_args.zero_noise = 0
    rollout_args.use_predicted_joints = 0
    rollout_args.dataset = 'babel'
    rollout_args.fix_floor = 1
        
    # TRY NOT TO MODIFY: seeding
    random.seed(rollout_args.seed)
    np.random.seed(rollout_args.seed)
    torch.manual_seed(rollout_args.seed)
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.deterministic = rollout_args.torch_deterministic
    device = torch.device(rollout_args.device if torch.cuda.is_available() else "cpu")
    rollout_args.device = device

    if rollout_args.denoiser_checkpoint is None or rollout_args.denoiser_checkpoint == '':
        rollout_args.denoiser_checkpoint = 'mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt'
    denoiser_args, denoiser_model, vae_args, vae_model = load_mld(rollout_args.denoiser_checkpoint, device)

    diffusion_args = denoiser_args.diffusion_args
    diffusion_args.respacing = rollout_args.respacing
    print('diffusion_args:', asdict(diffusion_args))
    diffusion = create_gaussian_diffusion(diffusion_args)

    # load initial seed dataset
    dataset = SinglePrimitiveDataset(cfg_path=vae_args.data_args.cfg_path,  # cfg path from model checkpoint
                                     dataset_path=vae_args.data_args.data_dir,  # dataset path from model checkpoint
                                     body_type=vae_args.data_args.body_type,
                                     sequence_path=f'./data/stand.pkl' if rollout_args.dataset == 'babel' else f'./data/stand_20fps.pkl',
                                     batch_size=rollout_args.batch_size,
                                     device=device,
                                     enforce_gender='male',
                                     enforce_zero_beta=1,
                                     )
    root_save_dir = Path(rollout_args.save_dir)

    if Path(rollout_args.text_prompt).exists():
        with open(rollout_args.text_prompt, 'r') as f:
            texts: list[list[str]] = json.load(f)
            texts = [','.join(text).strip() for text in texts]
            for i, text_prompt in enumerate(texts):
                print(f'Generating [{text_prompt}]')
                rollout_args.save_dir = root_save_dir / f'motion_{i}'
                rollout(text_prompt, denoiser_args, denoiser_model, vae_args, vae_model, diffusion, dataset, rollout_args)
    else:
        raise ValueError(f'Text prompt file {rollout_args.text_prompt} does not exist')

if __name__ == '__main__':
    main()