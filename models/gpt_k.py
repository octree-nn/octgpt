import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import copy
# from transformers import top_k_top_p_filtering
from tqdm import tqdm
from utils.utils import seq2octree, sample
from models.octformer import OctFormer, SinPosEmb
from utils.distributed import get_rank
logger = logging.getLogger(__name__)

class GPT(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self,
                 n_embed=256,
                 n_head=8,
                 n_layer=8,
                 num_classes=1,
                 num_pred_tokens=2,
                 split_size=2,
                 vq_size=128,
                 embed_drop=0.1,
                 **kwargs):
        super(GPT, self).__init__()
        self.n_embed = n_embed
        self.n_layer = n_layer

        self.pos_emb = SinPosEmb(n_embed)

        self.split_emb = nn.Embedding(split_size, n_embed)
        self.class_emb = nn.Embedding(num_classes, n_embed)

        self.drop = nn.Dropout(embed_drop)
        self.blocks = OctFormer(channels=n_embed, num_blocks=n_layer, num_heads=n_head,
                                patch_size=4096, dilation=2, num_pred_tokens=num_pred_tokens, nempty=False, use_checkpoint=True)

        self.ln_x = nn.LayerNorm(n_embed)
        self.split_head = nn.Linear(n_embed, split_size)
        self.vq_head = nn.Linear(n_embed, vq_size)

        self.apply(self._init_weights)
        logger.info("number of parameters: %e", sum(p.numel()
                    for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, split, octree_in, depth_low, depth_high, category=None, vqvae=None):
        targets = copy.deepcopy(split)

        batch_size = octree_in.batch_size

        if category == None:
            category = torch.zeros(batch_size).long().to(split.device)
        cond = self.class_emb(category)  # 1 x C

        # positional embedding
        position_embeddings = self.pos_emb(
            octree_in, depth_low, depth_high)  # S x C

        x_token_embeddings = self.split_emb(split)  # S x C

        if vqvae is not None:
            nnum_vq = octree_in.nnum[depth_high]
            with torch.no_grad():
                vq_code = vqvae.extract_code(octree_in)
                zq, indices, _ = vqvae.quantizer(vq_code)
            x_token_embeddings = torch.cat([x_token_embeddings, zq], dim=0)
            targets = torch.cat([targets, indices], dim=0)

        embeddings = torch.cat(
            [cond, x_token_embeddings], dim=0)[:-1]  # (1+S) x C
        embeddings = embeddings + \
            position_embeddings[:embeddings.shape[0]]  # S x C

        # embeddings = embeddings.unsqueeze(0)
        x = self.drop(embeddings)

        x, presents = self.blocks(x, octree_in, depth_low, depth_high)
        x = self.ln_x(x)

        output = {}
        if vqvae is not None:
            split_logits = self.split_head(x[:-nnum_vq])
            vq_logits = self.vq_head(x[-nnum_vq:])
            output['split_loss'] = F.cross_entropy(
                split_logits, targets[:-nnum_vq])
            output['vq_loss'] = F.cross_entropy(
                vq_logits, targets[-nnum_vq:])
        else:
            split_logits = self.split_head(x)
            output['split_loss'] = F.cross_entropy(
                split_logits, targets)
            output['vq_loss'] = torch.tensor(0.0).to(split.device)

        # octree_out = ocnn.octree.init_octree(6, 4, 1, x.device)
        # octree_out = seq2octree(octree_out, logits.argmax(dim=1), depth_low, depth_high)

        return output

    @torch.no_grad()
    def generate(self, octree, depth_low, depth_high, category=None, vqvae=None):
        if category == None:
            category = torch.zeros(1).long().to(octree.device)
        cond = self.class_emb(category)  # 1 x C

        token_embeddings = cond  # 1 x C
        split = torch.tensor([], device=octree.device).long()
        vq_indices = torch.tensor([], device=octree.device).long()

        past = torch.empty(
            (self.n_layer, 0, self.n_embed * 3), device=octree.device)
        # past = None
        for d in range(depth_low, depth_high + 1):
            # if not need to generate vq code
            if d == depth_high and vqvae == None:
                break

            position_embeddings = self.pos_emb(
                octree, octree.full_depth, d)  # S x C
            nnum_d = octree.nnum[d]

            for i in tqdm(range(nnum_d)):
                embeddings = token_embeddings + \
                    position_embeddings[:token_embeddings.shape[0], :]  # S x C
                if past is not None:
                    x = self.drop(embeddings[-1:])
                    x, presents = self.blocks(
                        x, octree, depth_low, d, past)  # B x S x C
                    past = torch.stack(presents, dim=0)
                else:
                    x = self.drop(embeddings)
                    x, _ = self.blocks(x, octree, depth_low, d)  # B x S x C
                x = self.ln_x(x)

                if d < depth_high:
                    split_logits = self.split_head(x[-1:, :])
                    ix = sample(split_logits)
                    split = torch.cat([split, ix], dim=0)
                    token_embeddings = torch.cat(
                        [token_embeddings, self.split_emb(ix)], dim=0)
                else:
                    vq_logits = self.vq_head(x[-1:, :])
                    ix = sample(vq_logits)
                    vq_indices = torch.cat([vq_indices, ix], dim=0)
                    with torch.no_grad():
                        token_embeddings = torch.cat(
                            [token_embeddings, vqvae.quantizer.embedding(ix)], dim=0)
            if d < depth_high:
                octree = seq2octree(octree, split[-nnum_d:], d, d + 1)
                # utils.export_octree(
                    # octree, d + 1, f"mytools/octree/depth{d+1}/", index=get_rank())

        return octree, vq_indices
