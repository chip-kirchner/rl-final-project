"""Parser

"""
import argparse
from distutils.util import strtobool

def parse_args():
    parser = argparse.ArgumentParser()

    # Logging    
    parser.add_argument("--verbose", type=lambda x: bool(strtobool(x)), default=True, help="Log/print output")   
    parser.add_argument("--tb-log", type=lambda x: bool(strtobool(x)), default=False, help="Tensorboard log")    
    parser.add_argument("--wandb-log", type=lambda x: bool(strtobool(x)), default=False, help="Wandb log")
    parser.add_argument("--tag", type=str, default='DDRQN', help="Training tag (for run_id)")

    # Environment
    parser.add_argument("--env", type=str, default="multicapture", help="which environment to use")
    parser.add_argument("--seed", type=int, default=0, help="Experiment seed")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")
    parser.add_argument("--tot-episodes", type=int, default=10000, help="Total episodes")
    parser.add_argument("--norm-obs", type=lambda x: bool(strtobool(x)), default=False, help="Whether to norm obs")
    parser.add_argument("--norm-rew", type=lambda x: bool(strtobool(x)), default=False, help="Whether to norm rew")
    parser.add_argument("--n-agents", type=int, default=3, help="Number of players in environment")
    parser.add_argument("--partial-obs", type=lambda x: bool(strtobool(x)), default=True, help="Agent partial observability")
    parser.add_argument("--load-penalty", type=float, default=0.0, help="Penalty for agent trying to load without help")
    parser.add_argument("--step-penalty", type=float, default=0.0, help="Cost for taking one step in environment")
    parser.add_argument("--grid-size", type=int, default=6, help="N size of N x N grid")

    parser.add_argument("--coop", type=lambda x: bool(strtobool(x)), default=False, help="Agent cooeration")
    parser.add_argument("--player-lvl", type=int, default=3, help="Maximum player level")
    parser.add_argument("--n_food", type=int, default=2, help="Number of targets to place in map")

    parser.add_argument("--n_targets", type=int, default=1, help="number of targets for capture target")
    parser.add_argument("--target_capacity", type=int, default=3, help="number of players necessary to capture targets")
    parser.add_argument("--target_reward", type=int, default=10, help="reward for agents successfully capturing a target")
    

    # Algorithm 
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.995, help="Target smoothing coefficient")
    parser.add_argument("--eps-start", type=float, default=1.0, help="Initial eps value for random exploration")
    parser.add_argument("--eps-end", type=float, default=0.01, help="Final eps value for random exploration")
    parser.add_argument("--eps-frac", type=float, default=0.6, help="Fraction of `tot-episodes` to anneal eps")

    #Buffer
    parser.add_argument("--buffer-size", type=int, default=10000, help="Max number of episodes/sequences to store")  
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size of memory sample")

    # Update
    parser.add_argument("--lr", type=float, default=3e-4, help="The learning rate of the optimizer")
    parser.add_argument('--hysteretic', type=lambda x: bool(strtobool(x)), default=True, 
        help='Whether to use hysteretic lr decay or not')
    parser.add_argument('--hyst-start', type=float, default=0.2, help='Initial value of hysteretic learning rate')
    parser.add_argument('--hyst-end', type=float, default=0.4,  help='Final value of hysteretic learning rate')
    parser.add_argument("--delay-start", type=int, default=32, help="Number of episodes before starting training")
    parser.add_argument("--update-freq", type=int, default=10, help="Update every n° steps")

    # Network
    parser.add_argument("--h-size", type=int, default=64, help="the size of the dnn")
    parser.add_argument("--n_hidden", type=int, default=2, help="n° of hidden layers")
    
    # Metrics
    parser.add_argument("--last-n", type=int, default=100, help="Average metrics over this time horizon")

    # wandb
    parser.add_argument("--wandb-project-name", type=str, default="", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="", help="the entity (team) of wandb's project")
    parser.add_argument("--wandb-mode", type=str, default="offline", 
        help="online or offline wadb mode. if offline,, we'll try to sync immediately after the run")
    parser.add_argument("--wandb-code", type=lambda x: bool(strtobool(x)), default=False, 
        help="Save code in wandb")

    # Torch
    parser.add_argument("--n-cpus", type=int, default=4, help="N° of cpus/max threads for process")
    parser.add_argument("--th-deterministic", type=lambda x: bool(strtobool(x)), default=True, 
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, 
        help="if toggled, cuda will be enabled by default")

    #Mixer
    parser.add_argument("--mixer",  type=str, default="None", help="type of mixer to use")
    parser.add_argument("--mix_embed_dim", type=int, default=32, help="Embedded dim of mixing network")
    parser.add_argument("--hypernet_embed_dim", type=int, default=32, help="Hypernet embedded dim size")
    parser.add_argument("--hypernet_layers", type=int, default=2, help="Number of hypernet layers")
   
    args = parser.parse_args()
    return args