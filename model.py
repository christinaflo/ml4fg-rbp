import numpy as np
import torch
from torch import nn


class ConvTransformerBlock(nn.Module):
    def __init__(self, c_s, dropout, include_attn=True, include_conv=True, include_spatial_enc=False):
        super(ConvTransformerBlock, self).__init__()

        self.include_attn = include_attn
        self.include_conv = include_conv
        self.include_spatial_enc = include_spatial_enc

        if self.include_attn:
            if self.include_spatial_enc:
                self.linear_z = nn.Linear(
                    c_s, 4, bias=False
                )
            self.block = nn.TransformerEncoderLayer(d_model=c_s,
                                                    nhead=4,
                                                    dim_feedforward=2 * c_s,
                                                    activation='gelu',
                                                    batch_first=True,
                                                    norm_first=True,
                                                    dropout=dropout)
            self.dropout1 = nn.Dropout(dropout)

        if self.include_conv:
            self.norm = nn.LayerNorm(c_s)
            self.conv_block = nn.Sequential(
                nn.Conv1d(c_s, 2 * c_s, kernel_size=3, stride=1, padding='same'),
                nn.SiLU(),
                nn.Conv1d(2 * c_s, 4 * c_s, kernel_size=3, stride=1, padding='same'),
                nn.SiLU(),
                nn.Conv1d(4 * c_s, c_s, kernel_size=5, stride=1, padding='same'),
            )
            self.dropout2 = nn.Dropout(dropout)
            self.lin_out = nn.Linear(c_s, c_s)
            with torch.no_grad():
                self.lin_out.weight.fill_(0.0)
                self.lin_out.bias.fill_(0.0)

    def forward(
            self,
            s: torch.Tensor,
            zij: torch.Tensor,
            node_mask: torch.Tensor
    ):

        if self.include_attn:
            if self.include_spatial_enc:
                z = self.linear_z(zij)
                z = z.permute(0, 3, 1, 2)
                key_padding_mask = ~node_mask.to(torch.bool)
                src_mask = z.reshape(z.shape[0] * z.shape[1], z.shape[2], z.shape[3])
                s = self.block(src=s, src_mask=src_mask, src_key_padding_mask=key_padding_mask)
            else:
                s = self.block(src=s, src_key_padding_mask=node_mask)
        s = s + self.dropout2(self.lin_out(self.conv_block(self.norm(s).transpose(-2, -1)).transpose(-2, -1)))

        return s


class RNAModel(nn.Module):
    def __init__(self, dim, num_blocks_sep, num_blocks_joint, include_ss, ss_separate, include_graph):
        super(RNAModel, self).__init__()
        self.include_ss = include_ss
        self.ss_separate = ss_separate
        self.include_graph = include_graph

        self.pos_enc = PositionalEncoding1D(dim)
        self.seq_emb = nn.Embedding(5, dim)
        #self.centrality = nn.Embedding(16, dim)
        #self.z_emb = ConcatDiffEdgeEmbedding(c_s=dim, c_edge=1, c_z=dim)
        if self.include_graph:
            self.z_emb = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))

        if self.include_ss:
            self.struct_emb = nn.Embedding(6, dim)
            self.emb_out = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU())

            if self.ss_separate:
                self.ss_blocks = nn.ModuleList([ConvTransformerBlock(c_s=dim,
                                                                     dropout=0.2,
                                                                     include_attn=True,
                                                                     include_conv=True,
                                                                     include_spatial_enc=False)
                                                for _ in range(num_blocks_sep)])
                self.se_blocks = nn.ModuleList([ConvTransformerBlock(c_s=dim,
                                                                     dropout=0.2,
                                                                     include_attn=True,
                                                                     include_conv=True,
                                                                     include_spatial_enc=False)
                                                for _ in range(num_blocks_sep)])

        self.blocks = nn.ModuleList([ConvTransformerBlock(c_s=dim,
                                                          dropout=0.2,
                                                          include_attn=True,
                                                          include_conv=True,
                                                          include_spatial_enc=self.include_graph)
                                                for _ in range(num_blocks_joint)])

        self.out_block = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))

    def forward(self, seq, struct, edge_attr, mask):
        s = self.seq_emb(seq) * mask.unsqueeze(-1)
        s = s + self.pos_enc(s)
        #s = s + self.centrality(degrees)

        if self.include_graph:
            zij = self.z_emb(edge_attr)
        else:
            zij = edge_attr

        if self.include_ss:
            ss = self.struct_emb(struct) * mask.unsqueeze(-1)

            if self.ss_separate:
                ss = ss + self.pos_enc(ss)

                for block in self.se_blocks:
                    s = block(s=s, zij=zij, node_mask=mask)

                for block in self.ss_blocks:
                    ss = block(s=s, zij=zij, node_mask=mask)

            s = self.emb_out(torch.cat([s, ss], dim=-1)) * mask.unsqueeze(-1)

        for block in self.blocks:
            s = block(s=s, zij=zij, node_mask=mask)

        s = s * mask.unsqueeze(-1)
        s = torch.sum(s, dim=1)

        return self.out_block(s)


def get_emb(sin_inp):
    """
    Taken from: https://github.com/tatp22/multidim-positional-encoding
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding1D(nn.Module):
    """Taken from: https://github.com/tatp22/multidim-positional-encoding"""
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc


class ConcatDiffEdgeEmbedding(nn.Module):
    def __init__(self, c_s, c_edge, c_z):
        super(ConcatDiffEdgeEmbedding, self).__init__()

        self.c_z = c_z

        self.linear_tf_z_i = nn.Linear(c_s, c_z)
        self.linear_tf_z_j = nn.Linear(c_s, c_z)

        self.edge_emb = nn.Sequential(nn.Linear(c_z + c_edge, c_z, bias=False), nn.SiLU())

    def forward(
        self,
        s: torch.Tensor,
        aij: torch.Tensor
    ) -> torch.Tensor:
        h_emb_i = self.linear_tf_z_i(s)
        h_emb_j = self.linear_tf_z_j(s)

        pair_emb_diff = torch.abs(h_emb_i[..., None, :] - h_emb_j[..., None, :, :])

        if aij is not None:
            edge_input = torch.cat((pair_emb_diff, aij), dim=-1)
        else:
            edge_input = pair_emb_diff

        pair_emb = self.edge_emb(edge_input)

        return pair_emb
