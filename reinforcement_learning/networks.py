from torch import nn


def linear_q1():
    input_layer = nn.Linear(in_features=11, out_features=50)
    hidden_layer_1 = nn.Linear(in_features=50, out_features=300)
    hidden_layer_2 = nn.Linear(in_features=300, out_features=50)
    output_layer = nn.Linear(in_features=50, out_features=3)

    model = nn.Sequential(
        input_layer,
        nn.ReLU(),
        hidden_layer_1,
        nn.ReLU(),
        hidden_layer_2,
        nn.ReLU(),
        output_layer,
        nn.Softmax(dim=1)  # dim 0 is batch, dim 1 are values
    )
    return model


def linear_q2():
    input_layer = nn.Linear(in_features=11, out_features=120)
    output_layer = nn.Linear(in_features=120, out_features=3)
    dropout = nn.Dropout(0.15)

    model = nn.Sequential(
        input_layer,
        nn.ReLU(),
        dropout,
        output_layer,
        nn.Softmax(dim=1)  # dim 0 is batch, dim 1 are values
    )
    return model
