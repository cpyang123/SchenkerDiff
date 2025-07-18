import math
from functools import reduce

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor

import src.utils
from src.diffusion import diffusion_utils
from src.models.layers import Xtoy, Etoy, masked_softmax


class RotaryPositionalEmbeddings(nn.Module):
    """
    Custom implementation of Rotary Position Embedding (RoPE).
    """
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency matrix
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary positional embedding to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, n_heads, head_dim]
            input_pos: Position indices of shape [batch_size, seq_len]
            
        Returns:
            Tensor with rotary positional embedding applied
        """
        batch_size, seq_len, n_heads, head_dim = x.shape
        
        # Handle odd dimensions by only applying RoPE to even portion
        if head_dim % 2 != 0:
            # Split into even and odd parts
            x_even = x[..., :-1]  # All but last dimension
            x_odd = x[..., -1:]   # Last dimension
            head_dim_even = head_dim - 1
        else:
            x_even = x
            x_odd = None
            head_dim_even = head_dim
        
        # Skip RoPE if dimension is too small
        if head_dim_even < 2:
            return x
            
        # Split even portion into pairs for rotation
        x_pairs = x_even.view(batch_size, seq_len, n_heads, head_dim_even // 2, 2)
        
        # Create position encodings
        pos_seq = input_pos.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        freqs = torch.outer(pos_seq.flatten(), self.inv_freq[:head_dim_even//2])  # [batch_size * seq_len, dim//2]
        freqs = freqs.view(batch_size, seq_len, -1)  # [batch_size, seq_len, dim//2]
        
        cos_pos = torch.cos(freqs).unsqueeze(2).unsqueeze(-1)  # [batch_size, seq_len, 1, dim//2, 1]
        sin_pos = torch.sin(freqs).unsqueeze(2).unsqueeze(-1)  # [batch_size, seq_len, 1, dim//2, 1]
        
        # Apply rotation
        x1, x2 = x_pairs[..., 0], x_pairs[..., 1]  # [batch_size, seq_len, n_heads, dim//2]
        
        # Rotate: [cos, -sin; sin, cos] * [x1; x2]
        rotated_x1 = x1 * cos_pos.squeeze(-1) - x2 * sin_pos.squeeze(-1)
        rotated_x2 = x1 * sin_pos.squeeze(-1) + x2 * cos_pos.squeeze(-1)
        
        # Recombine pairs
        rotated_pairs = torch.stack([rotated_x1, rotated_x2], dim=-1)
        rotated_even = rotated_pairs.view(batch_size, seq_len, n_heads, head_dim_even)
        
        # Concatenate back with odd dimension if it exists
        if x_odd is not None:
            return torch.cat([rotated_even, x_odd], dim=-1)
        else:
            return rotated_even


class XEyTransformerLayer(nn.Module):
    """ Transformer that updates node, edge and global features
        d_x: node features
        d_e: edge features
        dz : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """
    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffX: int = 2048,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, **kw)

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.lin_y1 = Linear(dy, dim_ffy, **kw)
        self.lin_y2 = Linear(dim_ffy, dy, **kw)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.dropout_y1 = Dropout(dropout)
        self.dropout_y2 = Dropout(dropout)
        self.dropout_y3 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, X: Tensor, E: Tensor, y, node_mask: Tensor):
        """ Pass the input through the encoder layer.
            X: (bs, n, d)
            E: (bs, n, n, d)
            y: (bs, dy)
            node_mask: (bs, n) Mask for the src keys per batch (optional)
            Output: newX, newE, new_y with the same shape.
        """

        newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)

        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)

        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)

        new_y_d = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y_d)

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)

        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        ff_output_y = self.dropout_y3(ff_output_y)
        y = self.norm_y2(y + ff_output_y)

        return X, E, y


class NodeEdgeBlock(nn.Module):
    """ Self attention layer that also updates the representations on the edges. """
    def __init__(self, dx, de, dy, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q = Linear(dx, dx)
        self.k = Linear(dx, dx)
        self.v = Linear(dx, dx)

        # FiLM E to X
        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)

        # FiLM y to E
        self.y_e_mul = Linear(dy, dx)           # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, dx)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.y_y = Linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, node_mask):
        """
        :param X: bs, n, d        node features
        :param E: bs, n, n, d     edge features
        :param y: bs, dz           global features
        :param node_mask: bs, n
        :return: newX, newE, new_y with the same shape.
        """
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1

        # 1. Map X to keys and queries
        Q = self.q(X) * x_mask           # (bs, n, dx)
        K = self.k(X) * x_mask           # (bs, n, dx)
        diffusion_utils.assert_correctly_masked(Q, x_mask)
        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df

        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))

        Q = Q.unsqueeze(2)                              # (bs, 1, n, n_head, df)
        K = K.unsqueeze(1)                              # (bs, n, 1, n head, df)

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))
        diffusion_utils.assert_correctly_masked(Y, (e_mask1 * e_mask2).unsqueeze(-1))

        E1 = self.e_mul(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        E2 = self.e_add(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2                  # (bs, n, n, n_head, df)

        # Incorporate y to E
        newE = Y.flatten(start_dim=3)                      # bs, n, n, dx
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE

        # Output E
        newE = self.e_out(newE) * e_mask1 * e_mask2      # bs, n, n, de
        diffusion_utils.assert_correctly_masked(newE, e_mask1 * e_mask2)

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)    # bs, 1, n, 1
        attn = masked_softmax(Y, softmax_mask, dim=2)  # bs, n, n, n_head

        V = self.v(X) * x_mask                        # bs, n, dx
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
        V = V.unsqueeze(1)                                     # (bs, 1, n, n_head, df)

        # Compute weighted values
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)            # bs, n, dx

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V

        # Output X
        newX = self.x_out(newX) * x_mask
        diffusion_utils.assert_correctly_masked(newX, x_mask)

        # Process y based on X axnd E
        y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)               # bs, dy

        return newX, newE, new_y


class GraphTransformer(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU()):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims['X']
        self.out_dim_E = output_dims['E']
        self.out_dim_y = output_dims['y']

        # self.mlp_in_X = nn.Sequential(nn.Linear(input_dims['X'] , hidden_mlp_dims['X']), act_fn_in,
        #                               nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in)

        self.mlp_in_E = nn.Sequential(nn.Linear(input_dims['E'], hidden_mlp_dims['E']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in)

        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims['y'], hidden_mlp_dims['y']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in)
        
        # self.mlp_in_r = nn.Sequential(nn.Linear(input_dims['r'], hidden_mlp_dims['r']), act_fn_in,
        #                               nn.Linear(hidden_mlp_dims['r'], hidden_dims['dr']), act_fn_in)
        
        self.mlp_in_X_r = nn.Sequential(nn.Linear(input_dims['r'] + input_dims['X'], hidden_mlp_dims['X'] + hidden_mlp_dims['r']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['X'] + hidden_mlp_dims['r'], hidden_dims['dx']), act_fn_in)
        
        # self.mlp_in_X_r = nn.Sequential(nn.Linear(hidden_dims['dr'] +  hidden_dims['dx'], hidden_dims['dx']), act_fn_in)

        # Add RoPE for positional encoding
        # Apply to concatenated X_r features
        x_r_dim = input_dims['X'] + input_dims['r']
        self.rope = RotaryPositionalEmbeddings(
            dim=x_r_dim,  # dimension of concatenated X_r
            max_seq_len=4096,
            base=10000
        )

        self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=hidden_dims['dx'],
                                                            de=hidden_dims['de'],
                                                            dy=hidden_dims['dy'],
                                                            n_head=hidden_dims['n_head'],
                                                            dim_ffX=hidden_dims['dim_ffX'],
                                                            dim_ffE=hidden_dims['dim_ffE'])
                                        for i in range(n_layers)])

        self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['X'], output_dims['X']))

        self.mlp_out_E = nn.Sequential(nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['E'], output_dims['E']))

        self.mlp_out_y = nn.Sequential(nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['y'], output_dims['y']))

    def unnormalize_positions(self, normalized_pos):
        """Scale normalized positions to smallest integers preserving ratios."""
        batch_size, seq_len = normalized_pos.shape
        position_indices = torch.zeros_like(normalized_pos, dtype=torch.long)
        
        for b in range(batch_size):
            pos = normalized_pos[b]
            
            # Scale by precision factor and round to integers
            precision = 10000
            scaled = pos * precision
            int_positions = scaled.round().long()
            
            # Find GCD to reduce to smallest integers
            non_zero_positions = int_positions[int_positions > 0]
            if len(non_zero_positions) > 1:
                common_gcd = reduce(math.gcd, non_zero_positions.tolist())
                if common_gcd > 1:
                    int_positions = int_positions // common_gcd
            
            position_indices[b] = int_positions
        
        return position_indices

    def forward(self, X, E, r, y, node_mask): # add R Matrix here to represent Rythm
        bs, n = X.shape[0], X.shape[1]

        E_out = E.clone().detach()
        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X_to_out = X[..., :self.out_dim_X]
        E_to_out = E[..., :self.out_dim_E]
        y_to_out = y[..., :self.out_dim_y]

        # Extract and unnormalize positional encoding from 8th column of r
        normalized_pos = r[:, :, 7]  # Extract 8th column (0-indexed)
        position_indices = self.unnormalize_positions(normalized_pos)

        # X += self.mlp_in_r(r)
        # X_r = torch.cat((self.mlp_in_X(X), self.mlp_in_r(r)), dim=-1)
        X_r = torch.cat((X, r), dim = -1)

        # Apply RoPE to X_r before MLP processing
        # Reshape for RoPE: [bs, n, 1, x_r_dim] (treating as single head)
        X_r_rope_input = X_r.unsqueeze(2)
        X_r_with_rope = self.rope(X_r_rope_input, input_pos=position_indices)
        X_r = X_r_with_rope.squeeze(2)  # Back to [bs, n, x_r_dim]

        # if config.etc.use_r:
        #     # use r matrix
        # else:
        #     # don't

        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        after_in = src.utils.PlaceHolder(X=self.mlp_in_X_r(X_r), E=new_E, y=self.mlp_in_y(y)).mask(node_mask)
        X, E, y = after_in.X, after_in.E, after_in.y

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        y = self.mlp_out_y(y)

        X = (X + X_to_out)
        E = (E + E_to_out) * diag_mask
        y = y + y_to_out

        E = 1/2 * (E + torch.transpose(E, 1, 2))

        return src.utils.PlaceHolder(X=X, E=E_out, y=y).mask(node_mask)
