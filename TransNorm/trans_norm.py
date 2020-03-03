import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn import init
import torch

import itertools

import pprint

class _TransNorm(Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):

        super(_TransNorm, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))

        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean_source', torch.zeros(num_features))
            self.register_buffer('running_mean_target', torch.zeros(num_features))

            self.register_buffer('running_var_source', torch.ones(num_features))
            self.register_buffer('running_var_target', torch.ones(num_features))

            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        else:
            self.register_parameter('running_mean_source', None)
            self.register_parameter('running_mean_target', None)
            self.register_parameter('running_var_source', None)
            self.register_parameter('running_var_target', None)

        self.reset_parameters()


    def reset_parameters(self):

        if self.track_running_stats:

            self.running_mean_source.zero_()
            self.running_mean_target.zero_()
            self.running_var_source.fill_(1)
            self.running_var_target.fill_(1)

        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def forward(self, input, last_flag= False, option='residual', running_flag=False, kernel='Student'):
        self._check_input_dim(input)

        if self.training:
            batch_size = input.size()[0] // 2

            input_source = input[:batch_size]
            input_target = input[batch_size:]

            x_hat_source = F.batch_norm(input_source, \
                                        self.running_mean_source, \
                                        self.running_var_source, \
                                        self.weight, \
                                        self.bias, \
                                        self.training or not self.track_running_stats, \
                                        self.momentum, \
                                        self.eps) # momentum은 업데이트 할 때 쓰임

            x_hat_target = F.batch_norm(input_target, \
                                        self.running_mean_target, \
                                        self.running_var_target, \
                                        self.weight, \
                                        self.bias, \
                                        self.training or not self.track_running_stats, \
                                        self.momentum, \
                                        self.eps)

            x_hat = torch.cat((x_hat_source, x_hat_target), dim=0)

            if running_flag:
                weight = torch.abs(self.running_mean_source - self.running_mean_target)
                cur_mean_source = self.running_mean_source
                cur_mean_target = self.running_mean_target

            else:
                if input.dim() == 4:
                    input_source = input_source.permute(0,2,3,1).contiguous().view(-1, self.num_features)
                    input_target = input_target.permute(0,2,3,1).contiguous().view(-1, self.num_features)

                cur_mean_source = torch.mean(input_source, dim=0)
                cur_var_source = torch.var(input_source, dim=0)
                cur_mean_target = torch.mean(input_target, dim=0)
                cur_var_target = torch.var(input_target, dim=0)

            if kernel == 'Gaussian':
                weight = torch.abs(cur_mean_source - cur_mean_target)
                tau = torch.exp(-weight / (torch.median(weight) + self.eps))

            elif kernel == 'Softmax':
                weight = torch.abs(cur_mean_source - cur_mean_target)
                temperature = 0.05
                tau = nn.Softmax(dim=0)(weight/temperature)

            elif kernel == 'Student':
                weight = torch.abs(cur_mean_source / torch.sqrt(cur_var_source + self.eps) - \
                                   cur_mean_target / torch.sqrt(cur_var_target + self.eps))
                tau = 1.0 / (1.0 + weight)

            tau = self.num_features * tau / sum(tau)

            if input.dim() == 2 :
                tau = tau.view(1, self.num_features)
            elif input.dim() == 4 :
                tau = tau.view(1, self.num_featueres, 1, 1)

            if option == 'out':
                output = x_hat * tau.detach()
            elif option == 'None' :
                output = x_hat
            elif option == 'residual':
                output = x_hat * (1 + tau.detach())

            output_mean_source = torch.mean(output[:batch_size], dim = 0)
            output_mean_target = torch.mean(output[batch_size:], dim = 0)

            if last_flag:
                return output, tau, cur_mean_source, cur_mean_target, output_mean_source, output_mean_target
            else:
                return output

        else: # test mode 할 때 이부분 정리해야함 여기서 정리하기

            x_hat = F.batch_norm(input, \
                                 self.running_var_target, \
                                 self.weight, \
                                 self.bias, \
                                 self.training or not self.track_running_stats, \
                                 self.momentum, \
                                 self.eps)

            if kernel == 'Gaussian' :
                weight = torch.abs(self.running_mean_source - self.running_mean_target)
                tau = torch.exp(-weight / (torch.median(weight) + self.eps))

            elif kernel == "Softmax":
                weight = torch.abs(self.running_mean_source - self.running_mean_target)
                temperature = 0.5
                tau = nn.Softmax(dim=0)(weight/temperature)

            elif kernel == "Student":
                weight = torch.abs(self.running_mean_source / torch.sqrt(self.running_var_source + self.eps)
                                   - self.running_mean_target / torch.sqrt(self.running_var_target + self.eps))

                tau = 1.0 / (1.0 + weight)

            tau = self.num_features * tau / sum(tau)

            if input.dim() == 2:
                tau = tau.view(1, self.num_features)

            elif input.dim() == 4:
                tau = tau.view(1, self.num_features, 1,1)

            if option == 'out':
                output = x_hat * tau.detach()
            elif option == 'None':
                output = x_hat
            elif option == 'residual':
                output = x_hat * (1+tau.detach())
            return output

    def _check_input_dim(self, input):
        return NotImplemented

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs):

        version = metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        #print('===================state dict=====================')
        #print(state_dict.keys())
        #print('==================================================')
        self._load_from_state_dict_from_pretrained_model(state_dict,\
                                                         prefix,\
                                                         metadata,\
                                                         strict,\
                                                         missing_keys,\
                                                         unexpected_keys,\
                                                         error_msgs)


    def _load_from_state_dict_from_pretrained_model(self, state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs, start_type='warm'):
        # 기존의 batchnorm에는 없는 코드
        #print('-------------here 222')
        local_name_params = itertools.chain(self._parameters.items(), self._buffers.items())
        local_state = {k: v.data for k, v in local_name_params if v is not None}

        for name, param in local_state.items(): # items() 함수는 key, value 쌍 얻는 것이다
            key = prefix + name
            if key[-5:] == 'alpha':
                continue
            if key[-6:] == 'source' or key[-6:] == 'target':
                if start_type == 'cold':
                    continue
                elif start_type == 'warm':
                    key = key[:-7]
            if key in state_dict:
                input_param = state_dict[key]

                if input_param.shape != param.shape:
                    error_msgs.append('size mismatch for {}: copying a param of {} from checkpoint, '
                                      'where the shape is {} in current model.'
                                      .format(key, param.shape, input_param.shape))
                    continue
                if isinstance(input_param, Parameter):
                    input_param = input_param.data

                try:
                    param.copy_(input_param)
                except Exception:
                    error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}.'
                                      .format(key, param.size(), input_param.size()))
            elif strict:
                missing_keys.append(key)

    def extra_repr(self):
        return 'num_features = {num_features}, eps = {eps}, momentum={momentum}, affine={affine}, track_running_stats = {track_running_stats}'.format(**self.__dict__)





class TransNorm2d(_TransNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))


class TransNorm1d(_TransNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))