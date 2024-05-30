from base_block import *
from attention import *
from einops import repeat
from einops.layers.torch import Rearrange
from kan import KAN
    
class MDvT(nn.Module):
    def __init__(self, image_size, channels, num_classes, dim=64, kernels=[3, 3, 3], strides=[2, 2, 2],
                 heads=[2, 4, 4], depth=[2, 4, 6], pool='cls', dropout=0.1, emb_dropout=0.1, mqa = False, kan=False):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        self.dim = dim
        self.heads = heads
        ##### Stage 1 #######
        self.conv1, self.stage1_transformer, channels, image_size, dim = self.stage_block(
            channels, dim, 96, kernels[0], strides[0], (image_size + 1) // 2, depth[0], heads[0], mqa, kan)


        ##### Stage 2 #######
        self.conv2, self.stage2_transformer, channels, image_size, dim = self.stage_block(
            channels, dim, 120, kernels[1], strides[1], (image_size + 1) // 2, depth[1], heads[1], mqa, kan)

        ##### Stage 3 #######
        self.conv3 = InvertedResidual(channels, dim, 2, 1, 6)
        self.stage3_conv_embed = nn.Sequential(
            Rearrange('b c h w n-> b (h w n) c', h=(image_size + 1) // 2, w=(image_size + 1) // 2),
            nn.LayerNorm(dim)
        )
        self.stage3_transformer = Transformer(dim=dim, img_size=(image_size + 1) // 2, depth=depth[2],
                        heads=heads[2], dim_head=self.dim,
                        dropout=dropout, last_stage=True)


        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_large = nn.Dropout(emb_dropout)

        ##### Stage 4 #######
        head_type= nn.Linear(dim, num_classes) if not kan else KAN([dim,num_classes])
        self.head = PreNorm(dim, head_type)

    def stage_block(self, in_channels, out_channels, transformer_dim, kernels, strides, image_size, depth, heads, mqa, kan):
        
        conv = Conv3D(in_channels, out_channels, kernels, strides, 1, act=nn.ReLU())
        transformer = MobileViTBlock(out_channels, transformer_dim, (image_size + 1) // 2, depth, heads, mqa=mqa, kan=kan)

        in_channels = out_channels
        image_size = (image_size + 1) // 2
        dim = 2*out_channels
        return conv, transformer, in_channels, image_size, dim

    def forward(self, img):
        conv1 = self.conv1(img)
        xs_trans1 = self.stage1_transformer(conv1)

        conv2 = self.conv2(xs_trans1)
        xs_trans2 = self.stage2_transformer(conv2)

        conv3 = self.conv3(xs_trans2)
        xs_conv3 = self.stage3_conv_embed(conv3)
        b, n, _ = xs_conv3.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        xs_cat = torch.cat((cls_tokens, xs_conv3), dim=1)
        xs_drop = self.dropout_large(xs_cat)
        xs_trans3 = self.stage3_transformer(xs_drop)
        xs = xs_trans3.mean(dim=1) if self.pool == 'mean' else xs_trans3[:, 0]

        xs = self.head(xs)
        return xs
    
