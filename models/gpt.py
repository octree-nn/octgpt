import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import ocnn
import copy
# from transformers import top_k_top_p_filtering
from tqdm import tqdm
from utils import seq2octree
from models.octformer import OctFormer, SinPosEmb

logger = logging.getLogger(__name__)


def sample(logits, top_k=2, top_p=1.0, temperature=0.7):
    logits = logits[-1, :] / temperature
    probs = F.softmax(logits, dim=-1)

    # top_k = top_k
    # topk, indices = torch.topk(probs, k=top_k, dim=-1)
    # probs = torch.zeros(*probs.shape).to(probs.device).scatter_(1, indices, topk)

    # # top-p
    # top_p = top_p
    # sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    # cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # sorted_indices_to_remove = cumulative_probs > top_p

    # sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    # sorted_indices_to_remove[..., 0] = False

    # indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    # probs[indices_to_remove] = 0

    ix = torch.multinomial(probs, num_samples=1)
    return ix


class GPT(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, n_embed=256, n_head=8, n_layer=8, num_classes=1, vocab_size=2, embed_drop=0.1, **kwargs):
        super(GPT, self).__init__()
        self.n_embed = n_embed
        self.vocab_size = vocab_size

        # self.pos_emb = nn.Parameter(nn.Embedding(block_size, n_embed).weight[None])
        self.pos_emb = SinPosEmb(n_embed)

        self.token_emb = nn.Embedding(vocab_size, n_embed)

        self.class_enc = nn.Embedding(num_classes, n_embed)

        self.drop = nn.Dropout(embed_drop)
        # self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.blocks = OctFormer(channels=n_embed, num_blocks=n_layer, num_heads=n_head,
                                patch_size=4096, dilation=2, nempty=False, use_checkpoint=True)

        self.ln_x = nn.LayerNorm(n_embed)
        self.x_head = nn.Linear(n_embed, vocab_size, bias=False)

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

    def forward(self, x, octree, depth_low, depth_high, category=None):
        targets = copy.deepcopy(x)

        batch_size = octree.batch_size

        if category == None:
            category = torch.zeros(batch_size).long().to(x.device)
        cond = self.class_enc(category)  # 1 x C

        # positional embedding
        position_embeddings = self.pos_emb(
            octree, depth_low, depth_high)  # S x C

        x_token_embeddings = self.token_emb(x)  # S x C

        token_embeddings = torch.cat(
            [cond, x_token_embeddings], dim=0)  # (1+S) x C
        embeddings = token_embeddings[:-1] + position_embeddings  # S x C

        # embeddings = embeddings.unsqueeze(0)
        x = self.drop(embeddings)

        x = self.blocks(x, octree, depth_low, depth_high)

        logits = self.x_head(self.ln_x(x))
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size), targets.view(-1))
        # octree_out = ocnn.octree.init_octree(6, 4, 1, x.device)
        # octree_out = seq2octree(octree_out, logits.argmax(dim=1), depth_low, depth_high)

        return loss  # , octree_out

    @torch.no_grad()
    def generate(self, octree, depth_low, depth_high, tokens=None, category=None):
        if category == None:
            category = torch.zeros(1).long().to(octree.device)
        cond = self.class_enc(category)  # 1 x C

        for d in range(depth_low, depth_high):
            position_embeddings = self.pos_emb(
                octree, octree.full_depth, d)  # S x C
            nnum_d = octree.nnum[d]

            for i in tqdm(range(nnum_d)):
                if tokens is None:
                    embeddings = (cond + position_embeddings[:1, :])
                    x = self.drop(embeddings)
                    x = self.blocks(x, octree, depth_low, d)  # B x S x C
                    logits = self.x_head(self.ln_x(x))
                    ix = sample(logits)
                    tokens = ix
                else:
                    x_token_embeddings = self.token_emb(tokens)  # S x C

                    token_embeddings = torch.cat(
                        [cond,  x_token_embeddings], dim=0)  # (1+S) x C
                    embeddings = token_embeddings + \
                        position_embeddings[:token_embeddings.shape[0], :]  # S x C
                    # print(embeddings.shape)

                    x = self.drop(embeddings)
                    x = self.blocks(x, octree, depth_low, d)  # B x S x C
                    logits = self.x_head(self.ln_x(x))
                    ix = sample(logits)
                    tokens = torch.cat((tokens, ix), dim=0)
                # print(torch.where(tokens[4096:] != gt_tokens[4096:tokens.shape[0]])[0].shape)
            octree = seq2octree(octree, tokens[-nnum_d:], d, d + 1)

        return octree
