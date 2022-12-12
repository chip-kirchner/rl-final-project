import torch.nn as nn
import torch.nn.functional as F

from utils.misc import *

def Linear(input_dim, output_dim, act_fn='leaky_relu', init_weight_uniform=True):
    """
    Creat a linear layer.

    Parameters
    ----------
    input_dim : int
        The input dimension.
    output_dim : int
        The output dimension.
    act_fn : str
        The activation function.
    init_weight_uniform : bool
        Whether uniformly sample initial weights.
    """
    gain = th.nn.init.calculate_gain(act_fn)
    fc = th.nn.Linear(input_dim, output_dim)
    if init_weight_uniform:
        nn.init.xavier_uniform_(fc.weight, gain=gain)
    else:
        nn.init.xavier_normal_(fc.weight, gain=gain)
    nn.init.constant_(fc.bias, 0.00)
    return fc

class RQNetwork(nn.Module):
    def __init__(self, env, h_size: int, n_hidden: int):
        super().__init__()

        o_space, a_space = env.observation_space, env.action_set
        self.a_dim = len(a_space) + 1 #env doesn't include no-ops
        print(self.a_dim)
        self.h_size = h_size
        
        self.hidden_1 = Linear(int(np.prod(o_space[0].shape)), h_size)
        self.gru_1 = nn.GRU(h_size, h_size, batch_first=True)
        self.hidden_2 = Linear(h_size, h_size)
        self.output = Linear(h_size, self.a_dim, act_fn='linear')

    def forward(self, x, h=None):
        x = F.leaky_relu(self.hidden_1(x))
        x, h = self.gru_1(x, h)
        x = F.leaky_relu(self.hidden_2(x))
        x = self.output(x)
        return x, h

    def get_action(self, observation, h, eps=0.0):
        qvalues, h = self.forward(observation, h)
        if np.random.rand() < eps:
            return th.randint(high=self.a_dim, size=(1,)), h
        else:
            return th.argmax(qvalues, dim=-1), h
    
    def init_hidden(self):
        return th.zeros([1, self.h_size])
    
    def reset_hidden_state(self):
        return th.zeros(self.h_size)
