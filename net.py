import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs
class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3,
                              padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class ImpalaCNN(nn.Module):
    """
    Network from IMPALA paper implemented in ModelV2 API.
    Based on https://github.com/ray-project/ray/blob/master/rllib/models/tf/visionnet_v2.py
    and https://github.com/openai/baselines/blob/9ee399f5b20cd70ac0a871927a6cf043b478193f/baselines/common/models.py#L28
    """
    def __init__(self, state_shape, device='cpu'):
        super().__init__()
        self.device = device
        h, w, c = state_shape
        shape = (c, h, w)

        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.conv_seqs = nn.ModuleList(conv_seqs)
        self.hidden_fc = nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256)
        self.nn = nn.LSTM(input_size=256,hidden_size=256,num_layers=2,batch_first=True)

    def forward(self, s, state=None, seq_lens=None):
        if not isinstance(s, torch.Tensor):
            x = torch.tensor(s, device=self.device, dtype=torch.float)
        x = x / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)
        x = nn.functional.relu(x)
        x = self.hidden_fc(x)
        s = nn.functional.relu(x)
        if len(s.shape) == 2:
            bsz, dim = s.shape
            length = 1
        else:
            bsz, length, dim = s.shape
        s = s.view(bsz, length, -1)
        self.nn.flatten_parameters()
        if state is None:
            s, (h, c) = self.nn(s)
        else:
            # we store the stack data in [bsz, len, ...] format
            # but pytorch rnn needs [len, bsz, ...]
            s, (h, c) = self.nn(s, (state['h'].transpose(0, 1).contiguous(),
                                    state['c'].transpose(0, 1).contiguous()))
        logits = s[:, -1]
        return logits, {'h': h.transpose(0, 1).detach(),
                        'c': c.transpose(0, 1).detach()}


class Net(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape=0, device='cpu',
                 softmax=False):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(np.prod(state_shape), 256),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(256, 256), nn.ReLU(inplace=True)]
        if action_shape:
            self.model += [nn.Linear(256, np.prod(action_shape))]
        if softmax:
            self.model += [nn.Softmax(dim=-1)]
        self.model = nn.Sequential(*self.model)

    def forward(self, s, state=None, info={}):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        return logits, state


class Actor(nn.Module):
    def __init__(self, preprocess_net, action_shape):
        super().__init__()
        self.preprocess = preprocess_net
        self.last = nn.Linear(256, np.prod(action_shape))

    def forward(self, s, state=None, info={}):
        logits, h = self.preprocess(s, state)
        logits = F.softmax(self.last(logits), dim=-1)
        # logits = torch.argmax(logits)
        return logits, h

class ServerActor(nn.Module):
    def __init__(self, preprocess_net, action_shape):
        super().__init__()
        self.preprocess = preprocess_net
        self.last1 = nn.Linear(256, np.prod(action_shape))
        self.last2 = nn.Linear(256, np.prod(action_shape))
        self.last3 = nn.Linear(256, np.prod(action_shape))

    def forward(self, s, state=None, info={}):
        latent, h = self.preprocess(s, state)
        logits1 = F.softmax(self.last1(latent), dim=-1)
        logits2 = F.softmax(self.last2(latent), dim=-1)
        logits3 = F.softmax(self.last3(latent), dim=-1)
        logits = torch.stack((logits1,logits2,logits3),dim=1)
        # logits = torch.argmax(logits)
        return logits, h

class RelayActor(nn.Module):
    def __init__(self, preprocess_net, action_shape):
        super().__init__()
        self.preprocess = preprocess_net
        self.last1 = nn.Linear(256, np.prod(action_shape))
        self.last2 = nn.Linear(256, np.prod(action_shape))

    def forward(self, s, state=None, info={}):
        latent, h = self.preprocess(s, state)
        logits1 = F.softmax(self.last1(latent), dim=-1)
        logits2 = F.softmax(self.last2(latent), dim=-1)
        logits = torch.stack((logits1, logits2), dim=1)
        # logits = torch.argmax(logits)
        return logits, h

class NFActor(nn.Module):
    def __init__(self, preprocess_net, action_shape):
        super().__init__()
        self.preprocess = preprocess_net
        self.last1 = nn.Linear(256, np.prod(action_shape))
        self.last2 = nn.Linear(256, np.prod(action_shape))
        self.last3 = nn.Linear(256, np.prod(action_shape))

    def forward(self, s, state=None, info={}):
        latent, h = self.preprocess(s, state)
        logits1 = F.softmax(self.last1(latent), dim=-1)
        logits2 = F.softmax(self.last2(latent), dim=-1)
        logits3 = F.softmax(self.last3(latent), dim=-1)
        logits = torch.stack((logits1,logits2,logits3),dim=1)
        # logits = torch.argmax(logits)
        return logits, h

class Critic(nn.Module):
    def __init__(self, preprocess_net):
        super().__init__()
        self.preprocess = preprocess_net
        self.last = nn.Linear(256, 1)

    def forward(self, s, **kwargs):
        logits, h = self.preprocess(s, state=kwargs.get('state', None))
        logits = self.last(logits)
        return logits


class Recurrent(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape, device='cpu'):
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.fc1 = nn.Linear(np.prod(state_shape), 128)
        self.nn = nn.LSTM(input_size=128, hidden_size=128,
                          num_layers=layer_num, batch_first=True)
        self.fc2 = nn.Linear(128, np.prod(action_shape))

    def forward(self, s, state=None, info={}):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        # s [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        if len(s.shape) == 2:
            bsz, dim = s.shape
            length = 1
        else:
            bsz, length, dim = s.shape
        s = self.fc1(s.view([bsz * length, dim]))
        s = s.view(bsz, length, -1)
        self.nn.flatten_parameters()
        if state is None:
            s, (h, c) = self.nn(s)
        else:
            # we store the stack data in [bsz, len, ...] format
            # but pytorch rnn needs [len, bsz, ...]
            s, (h, c) = self.nn(s, (state['h'].transpose(0, 1).contiguous(),
                                    state['c'].transpose(0, 1).contiguous()))
        s = self.fc2(s[:, -1])
        # please ensure the first dim is batch size: [bsz, len, ...]
        return s, {'h': h.transpose(0, 1).detach(),
                   'c': c.transpose(0, 1).detach()}
