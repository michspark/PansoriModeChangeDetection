import torch

def filter_by_confidence(freq_conf_tensor, threshold=0.8):
  freq_conf_tensor = torch.clone(freq_conf_tensor)

  if freq_conf_tensor.ndim == 3:
    assert freq_conf_tensor.shape[1] == 2
    flattened_tensor = freq_conf_tensor.permute(0,2,1).flatten(0,1)
    flattened_tensor[flattened_tensor[:,1]<threshold, 0] = 0
    freq_conf_tensor = flattened_tensor.reshape(freq_conf_tensor.shape[0], freq_conf_tensor.shape[2], -1).permute(0,2,1)

  elif freq_conf_tensor.ndim==2:
    assert freq_conf_tensor.shape[0] == 2
    freq_conf_tensor[0, freq_conf_tensor[1,:]<threshold] = 0
    freq_conf_tensor = freq_conf_tensor.unsqueeze(0)

  else:
    raise NotImplementedError

  return freq_conf_tensor


def calc_conv_param(config):
    num_layers = config.num_layers
    parameters = [{'input_channel':0, 'output_channel':0, 'max_pool':0} for _ in range(num_layers)]

    pool_size = config.pool_size
    if pool_size:
      for param in parameters: param['max_pool'] = pool_size

    in_channels = config.in_channels
    last_hidden_dim = config.last_hidden_dim

    if config.use_gradual_size:
        for idx, param in enumerate(parameters[:num_layers-1], start=1):
            sacle_factor = 1/2**(num_layers-1-idx)
            param['output_channel'] = int(last_hidden_dim * sacle_factor)
            param['input_channel'] = int(last_hidden_dim * (sacle_factor/2))
        parameters[num_layers-1]['input_channel'] = last_hidden_dim
        parameters[num_layers-1]['output_channel'] = last_hidden_dim

    else:
        for param in parameters:
            param['output_channel'] = last_hidden_dim
            param['input_channel'] = last_hidden_dim

    parameters[0]['input_channel'] = in_channels

    return parameters