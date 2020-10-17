from fastai.torch_core import *
from fastai.layers import *
from fastai.layers import conv1d


def deep_conv(channels:int, n_layers:int=8, filters:Tuple[Callable, int, Collection[int]]=128,
              downsamples:Tuple[Callable, int, Collection[int]]=None, drop:float=None, ks:int=3, is_1d=True, **res_params):
    """

    >>> downsamples, num_filters = downsample_updown, partial(filters_ascending)
    >>> data = torch.rand(10, 100, 128)
    >>> model, filters = deep_conv(100, drop=0.1, dense=True, downsamples=downsamples, filters=num_filters)
    >>> model(data).size()

    :param channels:
    :param n_layers:
    :param num_filters:
    :param downsamples:
    :param drop:
    :param ks:
    :param is_1d:
    :param res_params:
    :return:
    """
    assert ks % 2 == 1
    strides = [1] * n_layers if downsamples is None else downsamples
    strides = strides(n_layers) if isinstance(strides, Callable) else strides
    filters = filters(n_layers) if isinstance(filters, Callable) else filters
    filters = [channels] + (filters // np.asarray(strides)).tolist()
    res_net = len(res_params) > 0

    res = []
    for i, (ni, nf, stride) in enumerate(zip(filters[:-1], filters[1:], strides)):
        conv_params = { 'ks': ks, 'bias': True, 'is_1d': is_1d, 'use_activ':True}
        if res_net is True:
            if ni != nf: res += [conv_layer(ni, nf, **conv_params)]
            layers = res_block(nf, **conv_params)
        else: layers = conv_layer(ni, nf, **conv_params)

        if i > 0 and drop is not None:
            dropout = nn.Dropout(drop, inplace=True)
            if res_net: layers.append(dropout)
            else: layers.add_module(str(len(layers)), dropout)
        res += [layers]
    return nn.Sequential(*res), filters[-1]


def filters_ascending(amount, num_start_filters:int=64, increase_per:int=None):
    if increase_per is None: increase_per = max(int(log(amount)), 1)
    return [2**int(index / increase_per) * num_start_filters for index in range(amount)]


def downsample_updown(count:int, stride:int=1, every:int=2, ratio:int=2):
    assert every > 1
    return [stride * ratio if (i + 1) % every == 0 and i != 0 else stride for i in range(count)]


class CategorizeLayer(nn.Module):
    def __init__(self, emb_szs:ListSizes, emb_drop:float=0.0):
        super().__init__()
        self.embeds = nn.ModuleList([embedding(ni, nf) for ni, nf in emb_szs])
        self.drop = nn.Dropout(emb_drop, inplace=True)

    @property
    def channels(self): 
        return sum(e.embedding_dim for e in self.embeds)

    def forward(self, x):
        """ x is in the shape (batch, steps, channels) """
        x = [e(x[:, :, i]) for i, e in enumerate(self.embeds)]
        x = torch.cat(x, dim=-1)
        x = self.drop(x)
        return x

class AdaptiveConcatPool1d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, sz:Optional[int]=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool1d(self.output_size)
        self.mp = nn.AdaptiveMaxPool1d(self.output_size)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


class AttentionSequnce(nn.Module):
    def __init__(self, *args, attention:bool=False, lstm=nn.LSTM, **kwargs):
        super().__init__()
        self.lstm = lstm(*args, **kwargs)
        if attention:
            num_layers = 1 if 'num_layers' not in kwargs else kwargs['num_layers']
            self.attention = nn.Sequential(
                Lambda(lambda x: x.permute(0, 2, 1)),
                SelfAttention(args[1]*num_layers),
                Lambda(lambda x: x.permute(0, 2, 1))
            )
        else: self.attention = None
    
    def forward(self, *input):
        x, h = self.lstm(*input)
        if self.attention is not None: x = self.attention(x)
        return x, h
    

class SequentialBatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
    
    def forward(self, x):
        """ x in the shape (batch, steps, channels)"""
        x = x.permute(0, 2, 1)
        x = self.bn(x.contiguous())
        x = x.permute(0, 2, 1)
        return x.contiguous()


class StepFeature(nn.Module):
    def __init__(self, ni:int, nf:int, sizes:Collection[int], ps:Collection[float]=None, use_bn:bool=True, bn_final:bool=False):
        super().__init__()
        ps = listify(ifnone(ps, [0]*len(sizes)), sizes)
        sizes = [ni] + sizes + [nf]
        actns = [nn.ReLU(inplace=True) for _ in range(len(sizes)-2)] + [None]
        layers = []
        for i,(n_in,n_out,dp,act) in enumerate(zip(sizes[:-1],sizes[1:],[0.]+ps,actns)):
            layers += self.bn_drop_lin(n_in, n_out, bn=use_bn and i!=0, p=dp, actn=act)
        if bn_final: layers.append(SequentialBatchNorm1d(sizes[-1]))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


    def bn_drop_lin(self, n_in:int, n_out:int, bn:bool=True, p:float=0., actn:Optional[nn.Module]=None):
        "Sequence of SequentialBatchNorm1d (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
        layers = [SequentialBatchNorm1d(n_in)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        layers.append(nn.Linear(n_in, n_out))
        if actn is not None: layers.append(actn)
        return layers


class SimpleSelfAttention(nn.Module):
    """ https://github.com/sdoria/SimpleSelfAttention """
    def __init__(self, n_in:int, n_out:int=None, ks=1, sym=False):
        super().__init__()
        self.conv = conv1d(n_in, n_in if n_out is None else n_out, ks, padding=ks//2, bias=False)
        self.gamma = nn.Parameter(tensor([0.]))
        self.sym = sym
        self.n_in = n_in
        
    def forward(self,x):
        if self.sym:
            # symmetry hack by https://github.com/mgrankin
            c = self.conv.weight.view(self.n_in, self.n_in)
            c = (c + c.t())/2
            self.conv.weight = c.view(self.n_in, self.n_in,1)
                
        size = x.size()
        x = x.view(*size[:2],-1)
        o = torch.bmm(x.permute(0,2,1).contiguous(),self.conv(x))       
        o = self.gamma * torch.bmm(x,o) + x

        return o.view(*size).contiguous()        