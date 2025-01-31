from plot_utils import *
import argparse
import torch
import os

from utils import set_seed, cal_subtb_coef_matrix, fig_to_image, get_gfn_optimizer, get_gfn_forward_loss, \
    get_gfn_backward_loss, get_exploration_std, get_name
from buffer import ReplayBuffer
from langevin import langevin_dynamics
from models import GFN
from gflownet_losses import *
from energies import *
from evaluations import *

import matplotlib.pyplot as plt
from tqdm import trange

parser = argparse.ArgumentParser(description='GFN Linear Regression')
parser.add_argument('--lr_policy', type=float, default=1e-3)
parser.add_argument('--lr_flow', type=float, default=1e-2)
parser.add_argument('--lr_back', type=float, default=1e-3)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--s_emb_dim', type=int, default=64)
parser.add_argument('--t_emb_dim', type=int, default=64)
parser.add_argument('--harmonics_dim', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--buffer_size', type=int, default=300 * 100 * 2)
parser.add_argument('--T', type=int, default=100)
parser.add_argument('--epochs', type=int, default=25000)
parser.add_argument('--subtb_lambda', type=int, default=2)
parser.add_argument('--t_scale', type=float, default=5.)
parser.add_argument('--log_var_range', type=float, default=4.)
parser.add_argument('--energy', type=str, default='vae')
parser.add_argument('--mode_fwd', type=str, default="tb", choices=('tb', 'tb-avg', 'db', 'subtb', 'cond-tb-avg'))
parser.add_argument('--mode_bwd', type=str, default="tb", choices=('tb', 'tb-avg', 'mle', 'cond-tb-avg'))
parser.add_argument('--both_ways', action='store_true', default=False)
parser.add_argument('--repeats', type=int, default=10)

# For local search
################################################################
parser.add_argument('--local_search', action='store_true', default=False)

# How many iterations to run local search
parser.add_argument('--max_iter_ls', type=int, default=200)

# How many iterations to burn in before making local search
parser.add_argument('--burn_in', type=int, default=100)

# How frequently to make local search
parser.add_argument('--ls_cycle', type=int, default=100)

# langevin step size
parser.add_argument('--ld_step', type=float, default=0.001)

# langevin temperature
parser.add_argument('--ld_beta', type=float, default=5.)

# Langevin dynamics schedule
parser.add_argument('--ld_schedule', action='store_true', default=False)
parser.add_argument('--target_acceptance_rate', type=float, default=0.574)
################################################################


# For replay buffer
################################################################
# high beta give steep priorization in reward prioritized replay sampling
parser.add_argument('--beta', type=float, default=1.)

# low rank_weighted give steep priorization in rank-based replay sampling
parser.add_argument('--rank_weight', type=float, default=1e-2)

# three kinds of replay training: random, reward prioritized, rank-based
parser.add_argument('--prioritized', type=str, default="rank", choices=('none', 'reward', 'rank'))
################################################################

# Stepwise scheduler
parser.add_argument('--scheduler', action='store_true', default=False)
parser.add_argument('--step_point', type=int, default=7000)

parser.add_argument('--bwd', action='store_true', default=False)
parser.add_argument('--exploratory', action='store_true', default=False)

parser.add_argument('--sampling', type=str, default="buffer", choices=('sleep_phase', 'energy', 'buffer'))
parser.add_argument('--langevin', action='store_true', default=False)
parser.add_argument('--langevin_scaling_per_dimension', action='store_true', default=False)
parser.add_argument('--conditional_flow_model', action='store_true', default=False)
parser.add_argument('--learn_pb', action='store_true', default=False)
parser.add_argument('--pb_scale_range', type=float, default=0.1)
parser.add_argument('--learned_variance', action='store_true', default=False)
parser.add_argument('--partial_energy', action='store_true', default=False)
parser.add_argument('--exploration_factor', type=float, default=0.1)
parser.add_argument('--exploration_wd', action='store_true', default=False)
parser.add_argument('--clipping', action='store_true', default=False)
parser.add_argument('--lgv_clip', type=float, default=1e2)
parser.add_argument('--gfn_clip', type=float, default=1e4)
parser.add_argument('--zero_init', action='store_true', default=False)
parser.add_argument('--pis_architectures', action='store_true', default=False)
parser.add_argument('--lgv_layers', type=int, default=3)
parser.add_argument('--joint_layers', type=int, default=2)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--weight_decay', type=float, default=1e-7)
parser.add_argument('--use_weight_decay', action='store_true', default=False)
parser.add_argument('--eval', action='store_true', default=False)
args = parser.parse_args()

set_seed(args.seed)
if 'SLURM_PROCID' in os.environ:
    args.seed += int(os.environ["SLURM_PROCID"])

eval_data_size = 300
final_eval_data_size = 300
plot_data_size = 16
final_plot_data_size = 16

if args.pis_architectures:
    args.zero_init = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
coeff_matrix = cal_subtb_coef_matrix(args.subtb_lambda, args.T).to(device)

def get_energy():
    if args.energy == 'vae':
        energy = VAEEnergy(device=device, batch_size=args.batch_size)
    elif args.energy == 'linreg':
        energy = LinearEnergy(device=device, batch_size=args.batch_size)
    else:
        return NotImplementedError
    return energy

def eval():
    name = get_name(args)
    if not os.path.exists(name):
        os.makedirs(name)

    energy = get_energy()

    gfn_model = GFN(energy.data_ndim, args.s_emb_dim, args.hidden_dim, args.harmonics_dim, args.t_emb_dim,
                    trajectory_length=args.T, clipping=args.clipping, lgv_clip=args.lgv_clip, gfn_clip=args.gfn_clip,
                    langevin=args.langevin, learned_variance=args.learned_variance,
                    partial_energy=args.partial_energy, log_var_range=args.log_var_range,
                    pb_scale_range=args.pb_scale_range,
                    t_scale=args.t_scale, langevin_scaling_per_dimension=args.langevin_scaling_per_dimension,
                    conditional_flow_model=args.conditional_flow_model, learn_pb=args.learn_pb,
                    pis_architectures=args.pis_architectures, lgv_layers=args.lgv_layers,
                    joint_layers=args.joint_layers, zero_init=args.zero_init, device=device, energy=args.energy).to(
        device)

    name = get_name(args)
    checkpoint = torch.load(f'{name}model.pt', weights_only=True)
    gfn_model.load_state_dict(checkpoint['model_state_dict'])
    eval_data = energy.sample(eval_data_size, evaluation=True).to(device)
    eval_data = eval_data[0].repeat(100, 1, 1)
    samples = gfn_model.sample(100, None, condition=eval_data)

    x = torch.linspace(-10, 10, 100)
    y = torch.linspace(-10, 10, 100)
    xx, yy = torch.meshgrid(x, y)
    states = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    condition = eval_data[0].repeat(states.shape[0], 1, 1)
    energy = energy.energy(states, condition)
    energy = energy.reshape(100, 100)

    fig, ax = plt.subplots(1,1, figsize=(10,6))
    ax.contourf(x, y, (-energy).exp().detach().numpy())
    ax.scatter(samples[:,0].detach().numpy(), samples[:,1].detach().numpy(), label="Amostras")
    ax.set_title("Amostras")
    plt.show()

if __name__ == '__main__':
    eval()
