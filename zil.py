import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pc_layer import PCLayer

random.seed(1)
np.random.seed(2)
torch.manual_seed(3)

# we measure divergence of different dtypes,
# so that we can see the small divergence is a result of rounding error
divergence_for_different_dtypes = {}

for dtype in [torch.float32, torch.float64]:

    torch.set_default_dtype(dtype)

    input_size = 64
    hidden_size = 32
    output_size = 10
    batch_size = 16
    p_lr = 0.001
    # meet C3
    x_lr = 1.0

    # this model is to get the same initialization for all following models
    # the index convention follows the NeurIPS-20 paper
    # keys of the dictionary is l
    model = {
        1: nn.Linear(hidden_size, output_size, bias=False),
        2: nn.Linear(hidden_size, hidden_size, bias=False),
        3: nn.Linear(input_size, hidden_size, bias=False),
    }

    # build both models
    model_bp = nn.Sequential(
        nn.Linear(input_size, hidden_size, bias=False),
        nn.Sigmoid(),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.Sigmoid(),
        nn.Linear(hidden_size, output_size, bias=False),
    )
    model_pc = nn.Sequential(
        nn.Linear(input_size, hidden_size, bias=False),
        PCLayer(),
        nn.Sigmoid(),
        nn.Linear(hidden_size, hidden_size, bias=False),
        PCLayer(),
        nn.Sigmoid(),
        nn.Linear(hidden_size, output_size, bias=False),
    )

    # a fake batch
    data, target = (
        torch.zeros(batch_size, input_size).normal_(0.0, 0.1),
        torch.zeros(batch_size, output_size).normal_(0.0, 0.1)
    )

    model_pc.train()
    model_pc(data)

    model_bp.train()
    model_pc.train()

    # the index convention follows the NeurIPS-20 paper
    # keys of the dictionary is l
    linear_layers_bp = {
        1: model_bp[4],
        2: model_bp[2],
        3: model_bp[0],
    }
    optimizer_p_bp = optim.SGD(
        model_bp.parameters(), lr=p_lr
    )

    # for pc models, optimizer_p and optimizer_x is built layer-wised,
    # thus, cannot managed by something like PCTrainer anymore
    # the index convention follows the NeurIPS-20 paper
    # keys of the dictionary is l
    linear_layers_pc = {
        1: model_pc[6],
        2: model_pc[3],
        3: model_pc[0],
    }
    pc_layers = {
        1: model_pc[4],
        2: model_pc[1],
    }
    optimizer_p_pc = {
        1: optim.SGD(
            linear_layers_pc[1].parameters(), lr=p_lr
        ),
        2: optim.SGD(
            linear_layers_pc[2].parameters(), lr=p_lr
        ),
        3: optim.SGD(
            linear_layers_pc[3].parameters(), lr=p_lr
        ),
    }
    optimizer_x_pc = {
        1: optim.SGD(
            [pc_layers[1].x], lr=x_lr
        ),
        2: optim.SGD(
            [pc_layers[2].x], lr=x_lr
        ),
    }

    # make sure the initial weights are the same for both models
    linear_layers_bp[1].weight.data.copy_(model[1].weight.data)
    linear_layers_bp[2].weight.data.copy_(model[2].weight.data)
    linear_layers_bp[3].weight.data.copy_(model[3].weight.data)
    linear_layers_pc[1].weight.data.copy_(model[1].weight.data)
    linear_layers_pc[2].weight.data.copy_(model[2].weight.data)
    linear_layers_pc[3].weight.data.copy_(model[3].weight.data)

    def loss_fn(output):
        return (output - target).pow(2).sum() * 0.5

    # train bp
    optimizer_p_bp.zero_grad()
    prediction = model_bp(data)
    loss = loss_fn(prediction)
    loss.backward()
    optimizer_p_bp.step()

    # naming convention of T
    # T = {
    #     t: {optimize_x: l, optimize_p: l,}
    # }
    T = {
        0: {'x': 1,    'p': 1},
        1: {'x': 2,    'p': 2},
        2: {'x': None, 'p': 3},
    }

    for t in T.keys():

        # meet C1
        if t == 0:
            pc_layers[1].is_sample_x = True
            pc_layers[2].is_sample_x = True
        else:
            pc_layers[1].is_sample_x = False
            pc_layers[2].is_sample_x = False

        optimizer_p_pc[1].zero_grad()
        optimizer_p_pc[2].zero_grad()
        optimizer_p_pc[3].zero_grad()
        optimizer_x_pc[1].zero_grad()
        optimizer_x_pc[2].zero_grad()

        prediction = model_pc(data)
        loss = loss_fn(prediction)
        energy = pc_layers[1].energy.sum() + \
            pc_layers[2].energy.sum()
        (energy + loss).backward()

        # meet C2
        if T[t]['p'] is not None:
            optimizer_p_pc[T[t]['p']].step()
        if T[t]['x'] is not None:
            optimizer_x_pc[T[t]['x']].step()

    divergence = []
    for layer_i in linear_layers_bp.keys():
        divergence.append(
            (
                linear_layers_bp[layer_i].weight.data -
                linear_layers_pc[layer_i].weight.data
            ).abs().sum()
        )
    divergence = torch.stack(divergence).sum()

    divergence_for_different_dtypes[
        str(dtype)
    ] = divergence.item()

print(
    '\nWith float32, the divergence is {}.\nWith float64, the divergence is {}.'
    '\nComparing the divergence of float64 to that of float32:'
    '\n(1), if it is reduced significantly, then the divergence is due to rounding errors;'
    '\n(2), if it is not reduced significantly, then the divergence is not due to rounding errors, but theoretically-existing divergence.'
    '\nTry unmeet one of C1, C2 and C3, say, set x_lr=0.99999 and re-run, you will see what it looks like for divergence to be a theoretically-existing divergence.'.format(
        divergence_for_different_dtypes['torch.float32'],
        divergence_for_different_dtypes['torch.float64'],
    )
)