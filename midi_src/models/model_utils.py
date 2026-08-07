
def calc_conv_param(config):
    num_layers = config.num_layers
    parameters = [{'input_channel': 0, 'output_channel': 0, 'max_pool': 0}
                  for _ in range(num_layers)]

    if config.pool_size:
        for param in parameters:
            param['max_pool'] = config.pool_size

    if config.use_gradual_size:
        for idx, param in enumerate(parameters[:num_layers - 1], start=1):
            scale = 1 / 2 ** (num_layers - 1 - idx)
            param['output_channel'] = int(config.last_hidden_dim * scale)
            param['input_channel']  = int(config.last_hidden_dim * (scale / 2))
        parameters[num_layers - 1]['input_channel']  = config.last_hidden_dim
        parameters[num_layers - 1]['output_channel'] = config.last_hidden_dim
    else:
        for param in parameters:
            param['output_channel'] = config.last_hidden_dim
            param['input_channel']  = config.last_hidden_dim

    parameters[0]['input_channel'] = config.in_channels

    return parameters
