from fastai.torch_core import *
from fastai.tabular.models import TabularModel
from layers import CategorizeLayer, AdaptiveConcatPool1d, AttentionSequnce, SequentialBatchNorm1d, deep_conv, filters_ascending, downsample_updown, SimpleSelfAttention
from fastai.layers import *
from data import DataProvider


class HistoricalModel(nn.Module):
    def __init__(self, hist_steps:int=7, in_feats:int=1024, out_feats:int=8, out_drop:float=0., attention:bool=False):
        super().__init__()
        assert out_feats % 2 == 0
        self.hist_steps = hist_steps
        self.steps = AttentionSequnce(in_feats, out_feats, batch_first=True, attention=attention)
        self.axis = AttentionSequnce(hist_steps+1, hist_steps+1, batch_first=True, attention=False)
        self.out = nn.Sequential(
            AdaptiveConcatPool1d(out_feats),
            SequentialBatchNorm1d(out_feats),
            nn.Dropout(out_drop),
            Flatten(),
        )
        apply_init(self.out, nn.init.kaiming_normal_)
    
    @property
    def channels(self):
        return self.out[0].output_size * 2 * (self.hist_steps + 1)

    def forward(self, x):
        """x in the shape (B, L, C)"""
        _, L, _ = x.size()
        padded = torch.stack([x[:, 0, :]]*self.hist_steps, dim=1)
        padded = torch.cat([x, padded], dim=1)
        hist =[]
        for i in range(self.hist_steps, L + self.hist_steps):
            step = padded[:, i - self.hist_steps: i+1, :]
            hist += [self.forward_step(step)]
        return torch.stack(hist, dim=1)

    def forward_step(self, x):
        """ x in the shape (B, hist_steps+1, C)"""
        steps, _ = self.steps(x)
        axes, _  = self.axis(x.permute(0, 2, 1))
        axes     = axes.permute(0, 2, 1)
        x        = torch.cat([steps, axes + x], dim=-1)
        x        = self.out(x)
        return x

class StericNet(nn.Module):
    
    def __init__(self, ch_times:int=10, ch_conts:int=10, cat_emb_szs:ListSizes=None, cat_emb_drop:float=0.0, c:int=2, 
                        hist_steps:int=7, hist_feats:int=1024, hist_attention:bool=False, atttention=SimpleSelfAttention,
                        num_start_filters=128, n_layers=4, downsamples=downsample_updown, m_drop=0.2,
                        out_feats:int=512):                        
        super().__init__()
        self.ch_times, self.ch_conts = ch_times, ch_conts
        self.embeds = CategorizeLayer(emb_szs=cat_emb_szs, emb_drop=cat_emb_drop)
        self.hist_model = HistoricalModel(hist_steps=hist_steps, in_feats=ch_times, out_feats=hist_feats, attention=hist_attention)

        filters=partial(filters_ascending, num_start_filters=num_start_filters, increase_per=None)
        self.model, model_channels = deep_conv(self.channels, n_layers=n_layers, drop=m_drop, downsamples=downsamples, filters=filters, dense=False)
        self.attention = atttention(model_channels)
        self.out = nn.Sequential(
            nn.Linear(model_channels, 2),
            Lambda(lambda x: x.permute(0, 2, 1))
        )
        apply_init(self.out, nn.init.kaiming_normal_)


    def forward(self, times, conts, cats):
        """
        times, conts, cats are all in the shape of (batch, seq_len, channels)
        """
        cats = self.embeds(cats)
        hist = self.hist_model(times)
        x = torch.cat([hist, conts, cats], dim=-1) # (batch, steps, channels)
        x = x.permute(0, 2, 1).contiguous()
        x = self.model(x)
        x = self.attention(x)
        x = x.permute(0, 2, 1)

        return self.out(x.contiguous())

    @property
    def channels(self):
        return self.hist_model.channels + self.ch_conts + self.embeds.channels


def build_model(db: DataProvider):
    (time_fields, cont_fields, cat_fields), cat_sizes = db.data_fields, db.cat_sizes
    model = StericNet(cat_emb_szs=cat_sizes, 
                      ch_times=len(time_fields), 
                      cat_emb_drop=0.05,
                      ch_conts=len(cont_fields), 
                      num_start_filters=32, 
                      n_layers=8, 
                      downsamples=None, 
                      m_drop=0.1,
                      atttention=SimpleSelfAttention,
                      hist_steps=7,
                      hist_feats=64)
    return model