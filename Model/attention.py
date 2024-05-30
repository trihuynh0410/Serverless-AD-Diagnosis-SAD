import numpy as np

from base_block import *
from kan import *

class ConvAttention(nn.Module):
    def __init__(self, dim, img_size, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 last_stage=False):

        super().__init__()
        self.last_stage = last_stage
        self.img_size = img_size
        self.heads = heads
        self.scale = dim_head ** -0.5

        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        pad = (kernel_size - q_stride)//2

        self.to_q = SepConv3d(dim, inner_dim, kernel_size, q_stride, pad, dim, use_norm_point=False)
        self.to_k = SepConv3d(dim, inner_dim, kernel_size, q_stride, pad, dim, use_norm_point=False)
        self.to_v = SepConv3d(dim, inner_dim, kernel_size, q_stride, pad, dim, use_norm_point=False)

        self.to_out = nn.Sequential(
            MutableLinear(inner_dim, dim),
            MutableDropout(dropout)
        ) if project_out else MutableIdentity()

    def rearrange_and_concat(self, transform, x, h, cls_token):
        x = transform(x)
        x = rearrange(x, 'b (h d) l w i-> b h (l w i) d', h=h)
        if self.last_stage:
            x = torch.cat((cls_token, x), dim=2)
        return x
    
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        if self.last_stage:
            cls_token = x[:, 0]
            x = x[:, 1:]
            cls_token = rearrange(cls_token.unsqueeze(1), 'b n (h d) -> b h n d', h = h)

        x = rearrange(x, 'b (l w i) n -> b n l w i', l=self.img_size, w=self.img_size)

        q = self.rearrange_and_concat(self.to_q, x, h, cls_token)
        k = self.rearrange_and_concat(self.to_k, x, h, cls_token)
        v = self.rearrange_and_concat(self.to_v, x, h, cls_token)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
    

class MultiHeadAttention(nn.Module):
    """
    This layer applies a multi-head self- or cross-attention as described in
    `Attention is all you need <https://arxiv.org/abs/1706.03762>`_ paper
    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`
        num_heads (int): Number of heads in multi-head attention
        attn_dropout (float): Attention dropout. Default: 0.0
        bias (bool): Use bias or not. Default: ``True``
    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float = 0.1,
        bias: bool = True,
        kan: bool = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        # if embed_dim % num_heads != 0:
        #     raise ValueError(
        #         "Embedding dim must be divisible by number of heads in {}. Got: embed_dim={} and num_heads={}".format(
        #             self.__class__.__name__, embed_dim, num_heads
        #         )
        #     )
        if not kan:
            self.qkv_proj = MutableLinear(in_features=embed_dim, out_features=3 * embed_dim, bias=bias)

            self.attn_dropout = MutableDropout(p=attn_dropout)
            self.out_proj = MutableLinear(in_features=embed_dim, out_features=embed_dim, bias=bias)
        else:
            self.qkv_proj = MutableKAN([embed_dim,3,4,5,3*embed_dim], base_activation=HSwish)

            self.attn_dropout = MutableDropout(p=attn_dropout)
            self.out_proj = MutableKAN([embed_dim,3,4,5,embed_dim], base_activation=HSwish)            

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward(self, x_q: torch.Tensor) -> torch.Tensor:
        # [N, P, C]
        b_sz, n_patches, in_channels = x_q.shape

        # self-attention
        # [N, P, C] -> [N, P, 3C] -> [N, P, 3, h, c] where C = hc
        qkv = self.qkv_proj(x_q).reshape(b_sz, n_patches, 3, self.num_heads, -1)

        # [N, P, 3, h, c] -> [N, h, 3, P, C]
        qkv = qkv.transpose(1, 3).contiguous()

        # [N, h, 3, P, C] -> [N, h, P, C] x 3
        query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        query = query * self.scaling

        # [N h, P, c] -> [N, h, c, P]
        key = key.transpose(-1, -2)

        # QK^T
        # [N, h, i, c] x [N, h, c, d] -> [N, h, i, d] (i = d = P, make it difference for using einsum)
        attn = torch.einsum('nhic,nhcd->nhid', query, key)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [N, h, P, P] x [N, h, P, c] -> [N, h, P, c]
        out = torch.einsum('nhpp,nhpc->nhpc', attn, value)

        # [N, h, P, c] -> [N, P, h, c] -> [N, P, C]
        out = out.transpose(1, 2).reshape(b_sz, n_patches, -1)
        out = self.out_proj(out)

        return out

class ProjWrapper(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, bias, use_norm, kan):
        super().__init__()
        if kan:
            self.proj_func = MutableKAN([in_channel, in_channel//2, out_channel])
        else:
            self.proj_func = Conv3D(in_channel, out_channel, kernel_size=kernel_size, padding=padding, bias=bias, use_norm=use_norm)
        self.condition = kan

    def forward(self, x):
        if self.condition:
            x = self.to_last_dim(x)
            # print("before pj func",x.shape)
        x = self.proj_func(x)
        # print("after pj func", x.shape)
        if self.condition:
            x = self.to_first_dim(x)
            # print("after to first dim",x.shape)
        return x

    @staticmethod
    def to_last_dim(t):
        num_dims = len(t.shape)
        permute_order = [0] + list(range(2, num_dims)) + [1]
        return t.permute(*permute_order)
    
    @staticmethod
    def to_first_dim(t):
        num_dims = len(t.shape)
        permute_order = [0, num_dims-1] + list(range(1, num_dims-1))
        return t.permute(*permute_order)

class MQAWithDownSampling(nn.Module):

    def __init__(
        self, embed_dim: int, num_heads: int, kv_dim: int,  
        query_strides: Optional[Union[int, Tuple[int, int, int]]] = 2, 
        kv_strides: Optional[Union[int, Tuple[int, int, int]]] = 2, 
        dw_kernel_size: Union[int, Tuple[int, int, int]]=3, 
        dropout: float = 0.,  norm_momentum: int = 0.99, norm_epsilon: int = 0.001,
        bias: bool = True, kan: bool=False
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"Embedding dim must be divisible by number of heads in {self.__class__.__name__}. Got: embed_dim={embed_dim} and num_heads={num_heads}")

        if query_strides > 1:
            self.query_downsampling = MutableAvgPool3d(kernel_size=query_strides)
            self.query_downsampling_norm = MutableBatchNorm3d(num_features=embed_dim, momentum=norm_momentum, eps=norm_epsilon)
            
        if kv_strides > 1:
            self.kv_dw_conv_norm = Conv3D(embed_dim, embed_dim, dw_kernel_size, kv_strides, bias=bias, use_norm=True)
            
        self.query_proj = ProjWrapper(embed_dim, num_heads*kv_dim, kernel_size=1, padding='valid', bias=bias, use_norm=False, kan=kan)
        self.kv_proj = ProjWrapper(embed_dim, kv_dim, kernel_size=1, padding='same', bias=bias, use_norm=False, kan=kan)
        self.output_proj = ProjWrapper(num_heads*kv_dim, embed_dim, kernel_size=1, padding='valid', bias=bias, use_norm=False, kan=kan)  
                   
        if query_strides > 1:
            self.upsampling = MutableUpsample(scale_factor=query_strides, mode='trilinear', align_corners=True)
        
        self.dropout = MutableDropout(p=dropout)
        self._query_strides = query_strides
        self._kv_strides = kv_strides
        self._num_heads = num_heads
        self._kv_dim = kv_dim
        self._kan=kan
    
    def reshape_input(self, t):
        s = t.shape
        num = torch.prod(torch.tensor(s[2:]))  
        return t.reshape(s[0], s[1], num.item()) 

    def reshape_projected_query(self, t, num_heads, key_dim, px):
        """Reshapes projected query: [b, h x k, n, n, n] -> [b, h, k, n x n x n]."""
        s = t.shape
        return t.reshape(s[0], num_heads, key_dim, px*px*px)
     
    def get_pixels(self, t):
        s = t.shape
        return s[2]

    def reshape_output(self, t, num_heads, px):
        """Reshape output:[b, h, k, n x n x n] -> [b, h x k, n, n, n]."""
        s = t.shape
        last_dim = s[2] * num_heads
        return t.reshape(s[0], last_dim, px, px, px)
            
    def forward(self, x):
        # print("x shape is",x.shape)
        # x shape: [b, c, n, n, n]
        px = self.get_pixels(x) # px shape: n

        if self._query_strides > 1:
            q = self.query_downsampling(x)
            q = self.query_downsampling_norm(q)

            # q shape: [b, c, n, n, n] -> [b, h x k, n, n, n]
            q = self.query_proj(q)

        else:
            q = self.query_proj(x)

        # q shape: [b, h x k, n, n, n] -> [b, h, k, n x n x n]
        q = self.reshape_projected_query(
            q,
            self._num_heads,
            self._kv_dim,
            px // self._query_strides,
        )

        if self._kv_strides > 1:
            k = self.kv_dw_conv_norm(x)
            k = self.kv_proj(k)
        else:
            k = self.kv_proj(x)

        # k shape: [b, k, p], p = m x m x m
        k = self.reshape_input(k)
        # print("k shape is",k.shape)
        # print("q shape is",q.shape)
        # desired q shape: [b, h, k, n x n x n]
        # desired k shape: [b, k, m x m x m]
        # desired logits shape: [b, n x n x n, h, m x m x m]
        logits = torch.einsum('bhkl,bkp->blhp', q, k)

        logits = logits / torch.sqrt(torch.tensor(self._kv_dim, dtype=x.dtype))

        attention_scores = self.dropout(torch.nn.functional.softmax(logits, dim=-1))
        if self._kv_strides > 1:
            v = self.kv_dw_conv_norm(x)
            v = self.kv_proj(v)
        else:
            v = self.kv_proj(x)

        # v shape: [b, k, p], p = m x m x m
        v = self.reshape_input(v)

        # desired attention shape: [b, n x n x n, h, m x m x m]
        # desired v shape: [b, k, m x m x m]
        # desired o shape: [b, h, k, n x n x n]
        o = torch.einsum('blhp,bkp->bhkl', attention_scores, v)

        # reshape o: [b, h, k, n x n x n] -> [b, h x k, n, n, n]
        o = self.reshape_output(
            o,
            self._num_heads,
            px // self._query_strides,
        )

        if self._query_strides > 1:
            o = self.upsampling(o)

        result = self.output_proj(o)
        assert result.shape == x.shape, "Output shape does not match input shape"

        return result
    
class Transformer(nn.Module):

    def __init__(self, dim, img_size, depth, heads, dim_head, dropout=0.5, last_stage=False, kan=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        mlp_dim = dim * 4
        
        for _ in range(depth):
            if last_stage:
                attention = ConvAttention(dim, img_size, heads=heads, dim_head=dim_head, dropout=dropout, last_stage=last_stage)
                ff = MutableKAN([dim, mlp_dim, dim], base_activation=HSwish) if kan else FeedForward(dim, mlp_dim, dropout=dropout, act_layer=HSwish())
            else:
                attention = MultiHeadAttention(dim, heads, kan=kan)
                ff = MutableKAN([dim, dim * 2, dim], base_activation=HSwish) if kan else FeedForward(dim, dim * 2, dropout=dropout, act_layer=HSwish())
            self.layers.append(nn.ModuleList([
                PreNorm(dim, attention),
                PreNorm(dim, ff)
            ]))
                        
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x
    
class MobileViTBlock(nn.Module):
    def __init__(self, in_channels, transformer_dim, img_size, depth=2, num_heads=4, 
                 dropout=0.1, patch_size=4, conv_ksize=3, last_stage=False, mqa=False, kan=False
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patch_area = patch_size ** 3
        self.mqa = mqa
        self.kan = kan
        self.local_rep = SepConv3d(in_channels, transformer_dim, conv_ksize, act=HSwish(), use_norm_point=False)

        global_rep = [Transformer(transformer_dim, img_size, depth, num_heads, dim_head=64, dropout=dropout, last_stage=last_stage, kan=kan), nn.LayerNorm(transformer_dim)]
        self.global_rep = MQAWithDownSampling(transformer_dim, num_heads, transformer_dim, kan=kan) if mqa else nn.Sequential(*global_rep)

        self.conv_proj = Conv3D(transformer_dim, in_channels, kernel_size=1, stride=1, act=HSwish())
        self.fusion = Conv3D(2 * in_channels, in_channels, conv_ksize, stride=1, act=HSwish())

    def unfold(self, x):
        batch_size, in_channels, orig_h, orig_w, orig_d = x.shape
        new_size = [int(math.ceil(d / self.patch_size) * self.patch_size) for d in x.shape[2:]]
        interpolate = new_size != list(x.shape[2:])
        if interpolate:
            x = F.interpolate(x, size=new_size, mode="trilinear", align_corners=False)

        num_patches = [d // self.patch_size for d in new_size]
        total_patches = np.prod(num_patches)
        x = x.reshape(x.shape[0] * self.patch_area, total_patches, -1)
        return x, {"orig_size": (orig_h, orig_w, orig_d), "batch_size": batch_size, "interpolate": interpolate, "total_patches": total_patches, "num_patches": num_patches}

    def fold(self, x, info_dict):
        x = x.reshape(info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1)
        x = x.reshape(info_dict["batch_size"], x.shape[-1], *np.array(info_dict["num_patches"]) * self.patch_size)
        if info_dict["interpolate"]:
            x = F.interpolate(x, size=info_dict["orig_size"], mode="trilinear", align_corners=False)
        return x

    def forward(self, x):
        res = x
        fm = self.local_rep(x)
        if not self.mqa:
            # convert feature map to patches
            patches, info_dict = self.unfold(fm)

            # learn global representations
            for transformer_layer in self.global_rep:
                patches = transformer_layer(patches)

            # [B x Patch x Patches x C] -> [B x C x Patches x Patch]
            fm = self.fold(x=patches, info_dict=info_dict)
        else:
            fm = self.global_rep(fm)
        fm = self.conv_proj(fm)
        output = self.fusion(torch.cat((res, fm), dim=1))
        # print("mobilevit dim", output.shape)
        # print("kan stage",self.kan)
        # print("mqa stage",self.mqa)
        return output