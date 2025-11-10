import torch.nn as nn
import torch
from torch import matmul
import torch.nn.functional as F
from functools import reduce
import math

  
class deconv(nn.Module):
    """
    Deconvolutional block for decoders
    """

    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        return self.conv(x)


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def __init__(self, p=0.1):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, query, key, value):
        scores = matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(query.size(-1))
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        p_val = matmul(p_attn, value)
        return p_val, p_attn


class Vec2Patch(nn.Module):
    def __init__(self, channel, hidden, output_size, kernel_size, stride, padding):
        super(Vec2Patch, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(hidden, c_out)
        self.to_patch = nn.Fold(output_size=output_size, kernel_size=kernel_size, stride=stride, padding=padding)
        h, w = output_size

    def forward(self, x):
        feat = self.embedding(x)
        b, n, c = feat.size()
        feat = feat.permute(0, 2, 1)
        feat = self.to_patch(feat)
        return feat


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, tokensize, d_model, head, mode, p=0.1):
        super().__init__()
        self.mode = mode
        self.query_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p=p)
        self.head = head
        self.h, self.w = tokensize

    def forward(self, q, k, v, t):
        assert q.size() == k.size() == v.size()
        bt, n, c = q.size() 
        b = bt // t
        c_h = c // self.head
        key = self.key_embedding(k)
        query = self.query_embedding(q)
        value = self.value_embedding(v)
        if self.mode == 's':
            key = key.view(b, t, n, self.head, c_h).permute(0, 1, 3, 2, 4)
            query = query.view(b, t, n, self.head, c_h).permute(0, 1, 3, 2, 4)
            value = value.view(b, t, n, self.head, c_h).permute(0, 1, 3, 2, 4)
            att, _ = self.attention(query, key, value)
            att = att.permute(0, 1, 3, 2, 4).contiguous().view(bt, n, c)
        elif self.mode == 't':
            key = key.view(b, t, 2, self.h//2, 2, self.w//2, self.head, c_h)
            key = key.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous().view(b, 4, self.head, -1, c_h)
            query = query.view(b, t, 2, self.h//2, 2, self.w//2, self.head, c_h)
            query = query.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous().view(b, 4, self.head, -1, c_h)
            value = value.view(b, t, 2, self.h//2, 2, self.w//2, self.head, c_h)
            value = value.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous().view(b, 4, self.head, -1, c_h)
            att, _ = self.attention(query, key, value)
            att = att.view(b, 2, 2, self.head, t, self.h//2, self.w//2, c_h)
            att = att.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(bt, n, c)
        elif self.mode =='st':
            key = key.view(b, t, self.h, self.w, self.head, c_h)
            key = key.permute(0, 4, 1, 2, 3, 5).contiguous().view(b, self.head, -1, c_h)
            query = query.view(b, t, self.h, self.w, self.head, c_h)
            query = query.permute(0, 4, 1, 2, 3, 5).contiguous().view(b, self.head, -1, c_h)
            value = value.view(b, t, self.h, self.w, self.head, c_h)
            value = value.permute(0, 4, 1, 2, 3, 5).contiguous().view(b, self.head, -1, c_h)
            att, _ = self.attention(query, key, value)
            att = att.view(b, self.head, t, self.h, self.w, c_h)
            att = att.permute(0, 2, 3, 4, 1, 5).contiguous().view(bt, n, c)
        output = self.output_linear(att)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, p=0.1):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        self.conv = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(p=p))

    def forward(self, x):
        x = self.conv(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, tokensize, hidden=128, num_head=4, mode='s', dropout=0.1):
        super().__init__()
        self.attention = MultiHeadedAttention(tokensize, d_model=hidden, head=num_head, mode=mode, p=dropout)
        self.ffn = FeedForward(hidden, p=dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, input):
        x, t = input['x'], input['t']
        x = self.norm1(x)
        x = x + self.dropout(self.attention(x, x, x, t))
        y = self.norm2(x)
        x = x + self.ffn(y)
        return {'x': x, 't': t}


class CNN_Encoder(nn.Module):
    """
    CNN Encoder block for spatial feature extraction
    """
    def __init__(self, in_ch=10, channel=256, hidden=512, kernel_size=(3, 3), stride=(3, 3), padding=(3, 3)):
        super(CNN_Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # self.patch2vec = nn.Conv2d(channel, hidden, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.encoder(x)
        # x = self.patch2vec(x)
        return x


class DecoderTransformer(nn.Module):
    """
    Transformer block in the decoder portion of a NLP transformer, adpated to spatial-temporal forecasting
    """

    def __init__(self, tokensize, hidden=128, num_head=4, mode='s', dropout=0.1):
        super().__init__()
        self.attention = MultiHeadedAttention(tokensize, d_model=hidden, head=num_head, mode=mode, p=dropout)
        self.ffn = FeedForward(hidden, p=dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x (supplies key and value), y (supplies query), t (time)
        kv, q, t = x['x'], x['y'], x['t']
        q = self.norm1(q)
        kv = self.norm1(kv)
        x = q + self.dropout(self.attention(q, kv, kv, t))
        y = self.norm2(x)
        x = x + self.ffn(y)
        return {'x': x, 't': t}


class LIF_DL(nn.Module):
    def __init__(self, in_ch=10, out_ch=1, channel=256, stack_num=2, num_head=4, dropout=0.):
        super().__init__()
        
        hidden = 512
        kernel_size = (3, 3)
        padding = (3, 3)
        stride = (3, 3)
        output_size = (32, 32)
        token_size = (12, 12)
        blocks = []

        # CNN Encoders 
        self.ice_encoder = CNN_Encoder(in_ch=3, channel=channel, hidden=hidden, kernel_size=kernel_size, stride=stride, padding=padding)
        self.ice_patch2vec = nn.Conv2d(channel, hidden, kernel_size=kernel_size, stride=stride, padding=padding)
        self.met_encoder = CNN_Encoder(in_ch=in_ch-3, channel=channel, hidden=hidden, kernel_size=kernel_size, stride=stride, padding=padding)
        self.met_patch2vec = nn.Conv2d(channel, hidden, kernel_size=kernel_size, stride=stride, padding=padding)
        
        # Transformer, one that combines sources, the rest are self attention
        # self.decoder_transformer = DecoderTransformer(token_size, hidden=hidden, num_head=num_head, mode='st', dropout=dropout)
        self.decoder_transformer = DecoderTransformer(token_size, hidden=hidden, num_head=num_head, mode='s', dropout=dropout)
        for _ in range(stack_num-1):
            blocks.append(TransformerBlock(token_size, hidden=hidden, num_head=num_head, mode='st', dropout=dropout))
        self.transformer = nn.Sequential(*blocks)
        self.vec2patch = Vec2Patch(channel, hidden, output_size, kernel_size, stride, padding)
               
        # CNN Decoder to return to the original dimension
        self.decoder = nn.Sequential(
            deconv(channel, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, out_ch, kernel_size=3, stride=1, padding=1)
        ) 

                    
    def forward(self, inputs):
        # Split input into the ice (initial states) and met (current states)
        initial_states = inputs[:,:,-3:,:,:]
        current_states = inputs[:,:,:-3,:,:]

        # Pass the initial timesteps to its encoder. Initial states are the ice
        b, t, c, h, w = initial_states.size()
        past_enc_features = self.ice_encoder(initial_states.view(b*t, c, h, w))
        past_trans_features = self.ice_patch2vec(past_enc_features)
        
        # Pass the current timesteps to its encoder. Current states are the coming met
        b, t, c, h, w = current_states.size()
        current_features = self.met_encoder(current_states.view(b*t, c, h, w))
        current_features = self.met_patch2vec(current_features)

        # Combine the two together
        # The decoder transformer block will use past features as key and value, 
        # while the future features get used as a query. 
        # with a skip connection for the future feature, 
        # this allows for an update of its state based on the previous information
        _, c, h, w = current_features.size()
        current_features = current_features.view(b*t, c, -1).permute(0, 2, 1)
        past_trans_features = past_trans_features.view(b*t, c, -1).permute(0, 2, 1)
        # x = self.decoder_transformer({'x':current_features, 'y':past_trans_features, 't':t})['x']
        x = self.decoder_transformer({'x':past_trans_features, 'y':current_features, 't':t})['x']

        # Now pass through the rest of the transformer blocks
        future_pred = self.transformer({'x':x, 't':t})['x']
        future_pred = self.vec2patch(future_pred)

        past_features = past_enc_features + future_pred

        # Finally pass to cnn decoder to cast back to original spatial domain and collapse channels
        output = self.decoder(future_pred)
        _, c, h, w = output.size()
        output = torch.sigmoid(output)

        return output.view(b, t, c, h, w)
