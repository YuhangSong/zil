"""

    An implementation of ZIL with pytorch

"""

import torch
import copy
import pprint
pp = pprint.PrettyPrinter(indent=4)

torch.manual_seed(3)

rule = "ZIL"
# rule = "BP"

L = 3
input_size = 32
hidden_size = 16
output_size = 10
batch_size = 8

inference_rate = 1.0

# sample a batch of fake data
data, target = (
    torch.zeros(
        batch_size, input_size
    ).normal_(0.0, 0.1),
    torch.zeros(
        batch_size, output_size
    ).scatter_(
        1, torch.LongTensor(
            batch_size, 1
        ).random_(
            0, output_size
        ), 1
    )
)

Ws = []

for l in range(L):

    if l == 0:

        input_dims = input_size
        output_dims = hidden_size

    elif l == L - 1:

        input_dims = hidden_size
        output_dims = output_size

    else:

        input_dims = hidden_size
        output_dims = hidden_size

    Ws.append(
        torch.zeros(
            input_dims, output_dims
        ).normal_(0.0, 0.1)
    )

Xs = []
MUs = []

for l in list(range(L)) + [L]:

    if l == 0:

        dims = input_size

    elif l == L:

        dims = output_size

    else:

        dims = hidden_size

    Xs.append(
        torch.zeros(
            batch_size, dims
        ).normal_(0.0, 0.1)
    )
    MUs.append(
        torch.zeros(
            batch_size, dims
        ).normal_(0.0, 0.1)
    )

Es = copy.deepcopy(Xs)

# forward

Xs[0].copy_(data)
Es[0].fill_(0.0)

for l in range(L):
    MUs[l + 1].copy_(torch.matmul(Xs[l], Ws[l]))
    Xs[l + 1].copy_(MUs[l + 1])
    Es[l + 1].copy_(Xs[l + 1]- MUs[l + 1])

# backward

Es[-1].copy_(target - Xs[-1])

if rule == "BP":

    for l in reversed(range(L)):

        Es[l].copy_(torch.matmul(Es[l + 1], Ws[l].t()))

elif rule == "ZIL":

    for l in reversed(range(L)):

        Xs[l] += inference_rate * torch.matmul(Es[l + 1], Ws[l].t())
        Es[l].copy_(Xs[l] - MUs[l])

# error on the input layer is not valid (it is not used or updated)
pp.pprint(Es[1:])