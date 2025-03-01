from typing import Callable, Optional, Union, Tuple, Any

import torch
from jaxtyping import Bool, Float, Int
import torch.utils

class Embedding(torch.nn.Module):

    positions: Int[torch.nn.UninitializedBuffer, "n_tokens"]

    def __init__(
        self,
        n_vocab: int,
        n_tokens: int = 512,
        dim_embed: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_length = n_tokens
        self.token = torch.nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=dim_embed,
            dtype=torch.bfloat16,
        )
        self.pos = torch.nn.Embedding(
            num_embeddings=n_tokens,
            embedding_dim=dim_embed,
            dtype=torch.bfloat16,
        )
        self.dropout = torch.nn.Dropout(p=dropout)
        self.register_buffer("positions", torch.arange(start=0,end=self.max_length,step=1,dtype=torch.int64))

    def forward(
        self, x: Float[torch.Tensor, "n_batch n_tokens"]
    ) -> Float[torch.Tensor, "n_batch n_tokens dim_embed"]:
        
        _, n_tokens = x.size()
        assert n_tokens <= self.max_length, f"Token sequence for batch exceeds max lenght of {self.max_length}"

        e_token: Float[
            torch.Tensor, "n_batch n_tokens dim_embed"
        ] = self.token(x)

        e_pos: Float[torch.tensor, "1 n_tokens dim_embed"] = self.pos(self.positions[:n_tokens]).unsqueeze(0)

        e = self.dropout(e_token + e_pos)

        return e

class MultiHeadedSelfAttention(torch.nn.Module):
    """Implements tensorized attention based on Andrej Karpaty's MinGPT"""

    causal_mask: Bool[torch.Tensor, "1 1 n_tokens n_tokens"]

    def __init__(
        self,
        n_tokens: int = 512,
        n_heads: int = 12,
        dim_embed: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert (
            dim_embed % n_heads == 0
        ), f"Query & value embedding size {dim_embed} is not divisable by "
        f"attention head count {n_heads}"

        self.max_length = n_tokens
        self.n_heads = n_heads
        self.dim_embed = dim_embed
        self.dim_sh_embed = int(dim_embed / n_heads)
        self.scale = torch.sqrt(torch.tensor(self.dim_sh_embed,requires_grad=False))
        self.register_buffer("causal_mask",torch.tril(torch.ones(1,1,n_tokens,n_tokens)).to(torch.bool))

        self.projection_attention = torch.nn.Linear(self.dim_embed, self.dim_embed * 3,dtype=torch.bfloat16,)
        self.dropout_attention = torch.nn.Dropout(p=dropout)
        self.projection_out = torch.nn.Linear(self.dim_embed,self.dim_embed,dtype=torch.bfloat16,)
        self.dropout_residual = torch.nn.Dropout(p=dropout)

    def forward(
            self,
            x: Float[torch.Tensor, "n_batch n_tokens dim_embed"],
        ) -> Float[torch.Tensor, "n_batch n_tokens dim_embed"]:

        n_batch, n_tokens, _ = x.size()
        assert n_tokens <= self.max_length, f"Token sequence for batch exceeds max lenght of {self.max_length}"

        # do all query, key and value projections for all heads in one go 
        # yielding 3 tensors of dimension 
        # (n_batch, n_tokens, dim_embed)
        q, k, v = self.projection_attention(x).split(self.dim_embed,dim=-1)

        # break down the query embedding into single head sized embeddings and
        # rearrange to add head dimension. This allows us to use transpose to 
        # generate tensor that effectively reduces complexity into generic 2d
        #  attention in its last two dimensions, i.e. we get a 4d Q tensor
        # (n_batch, n_head, n_tokens, n_dim_sh_embed)
        q_h = q.view(n_batch, n_tokens, self.n_heads, self.dim_sh_embed).transpose(1,2).contiguous()

        # apply equivalent transformation to keys to generate a 4d K tensor
        # (n_batch, n_head, n_tokens, n_dim_sh_embed)
        k_h = k.view(n_batch, n_tokens, self.n_heads, self.dim_sh_embed).transpose(1,2).contiguous()

        # batch matrix multiplication means we do a 
        # (n_tokens, n_dim_sh_embed) x (n_dim_sh_embed, n_tokens) matrix multiplication
        # for every fixed (batch sample, sequence token) pair in the first two
        # dimensions of our 4d tensors, giving us
        # (n_batch, n_head, n_tokens, n_tokens)
        k_h_T = k_h.transpose(-2,-1).contiguous()
        coef = (q_h @ k_h_T) / self.scale

        # before masking the coefficients, we add a head dimension to the mask
        #  to go from n_batch, 1, n_tokens, n_tokens 
        # to a compatible 4d tensor
        # (n_batch, 1, n_tokens, n_tokens)
        coef_masked = coef.masked_fill(self.causal_mask[:,:,:n_tokens,:n_tokens] == False,float('-inf'))

        conv_coef = torch.nn.functional.softmax(coef_masked,dim=-1)
        conv_coef = self.dropout_attention(conv_coef)

        # apply equivalent transformation to keys to generate a 4d V tensor
        # (n_batch, n_head, n_tokens, n_dim_sh_embed)
        v_h = v.view(n_batch, n_tokens, self.n_heads, self.dim_sh_embed).transpose(1,2).contiguous()

        # (n_batch, n_head, n_tokens, n_tokens) x (n_batch, n_head, n_tokens, n_dim_sh_embed) -> (n_batch, n_head, n_tokens, n_dim_sh_embed)
        att_h = conv_coef @ v_h

        # reshape into 3 dimensional format, implicitly concatenating the 
        # individual heads' attention outputs into one embedding vector for 
        # each token as required for the final step of multi head attention
        # (n_batch, n_tokens, n_dim_embed)
        att = att_h.transpose(1,2).contiguous().view(n_batch, n_tokens, self.dim_embed)

        att_proj = self.projection_out(att)

        return self.dropout_residual(att_proj)

class FeedForward(torch.nn.Module):
    """The feed forward layer (including dropout) as described in https://...
    ...cdn.openai.com/research-covers/language-unsupervised/...
    ...language_understanding_paper.pdf. Defaults to the exact configuration
     presented in the paper."""

    def __init__(
        self,
        dim_embed: int = 768,
        dim_ff: int = 3072,
        dropout: float = 0.1,
        activation_class=torch.nn.GELU,
    ):
        """By default, the GELU activation is used as described in the
        https://cdn.openai.com/research-covers/language-unsupervised/...
        ...language_understanding_paper.pdf paper.
        Choose nn.ReLU to implement the version described in the original
        transformer paper https://arxiv.org/pdf/1706.03762.

        Args:
            dim_input (int): The dimension of the input embedding vector space.
            dim_ff (int): The dimension of the hidden layer of this module.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
            activation_class (_type_, optional): Activation function class.
                Defaults to nn.GELU.
        """

        super().__init__()

        self.dim_embed = dim_embed
        self.dim_ff = dim_ff

        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(dim_embed, dim_ff,dtype=torch.bfloat16,),
            activation_class(),
            torch.nn.Linear(dim_ff, dim_embed,dtype=torch.bfloat16,),
            torch.nn.Dropout(p=dropout),
        )

    def forward(
        self, x: Float[torch.Tensor, "n_batch n_tokens dim_embed"]
    ) -> Float[torch.Tensor, "n_batch n_tokens dim_embed"]:
        return self.sequential(x)
    
class DecoderLayer(torch.nn.Module):

    def __init__(
        self,
        n_tokens: int = 512,
        n_heads: int = 12,
        dim_embed: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mha = MultiHeadedSelfAttention(
            n_tokens=n_tokens, n_heads=n_heads, dim_embed=dim_embed, dropout=dropout,
        )
        self.skip_norm_mha = torch.nn.LayerNorm(
            normalized_shape=(dim_embed,),dtype=torch.bfloat16,
        )
        self.ff = FeedForward(dim_embed=dim_embed, dropout=dropout)
        self.skip_norm_ff = torch.nn.LayerNorm(
            normalized_shape=(dim_embed,),dtype=torch.bfloat16,
        )

    def forward(
        self,
        x: Float[torch.Tensor, "n_batch n_tokens dim_embed"],
    ) -> Float[torch.Tensor, "n_batch n_tokens dim_embed"]:
        mha_out = self.skip_norm_mha(self.mha(x=x) + x)
        out = self.skip_norm_ff(self.ff(mha_out) + x)

        return out
        
class GPT(torch.nn.Module):

    def __init__(
        self,
        n_vocab: int,
        n_tokens=512,
        dim_embed: int = 768,
        n_decoder_layers: int = 12,
        n_heads: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = {
            "n_vocab":n_vocab,
            "n_tokens": n_tokens,
            "dim_embed": dim_embed,
            "n_decoder_layers": n_decoder_layers,
            "n_heads": n_heads,
            "dropout": dropout,
        }
        self.embedding = Embedding(
            n_vocab=n_vocab,
            n_tokens=n_tokens,
            dim_embed=dim_embed,
            dropout=dropout,
        )
        self.decoder = self.layers = torch.nn.Sequential(
            *[
                DecoderLayer(
                    n_heads=n_heads,
                    n_tokens=n_tokens,
                    dim_embed=dim_embed,
                    dropout=dropout,
                )
                for _ in range(n_decoder_layers)
            ]
        )

    def forward(
        self,
        x: Float[torch.Tensor, "n_batch n_tokens dim_embed"],
        target: Optional[Float[torch.Tensor, "n_batch n_tokens"]] = None,
    ) -> Union[Float[torch.Tensor, "n_batch n_tokens n_vocab"],Tuple[Float[torch.Tensor, "n_batch n_tokens n_vocab"],Float[torch.Tensor, "1"]]]:
        """We extend the core class' forward method by 
        - a language head that maps the last transformer layer's outputs back
            into the vocabulary space, and
        - an optional loss that is only invoked if the `target` arg is provided
        """
        e: Float[torch.Tensor, "n_batch n_tokens dim_embed"] = self.embedding(x)
        d: Float[torch.Tensor, "n_batch n_tokens dim_embed"] = self.decoder(e)
        logit: Float[torch.Tensor, "n_batch n_tokens n_vocab"] = torch.matmul(d,self.embedding.token.weight.transpose(-1,-2))

        if target is not None:
            # torch's cross entropy loss requires the vocabulary dimension in the
            # second rank as per https://pytorch.org/docs/stable/generated/...
            # ...torch.nn.CrossEntropyLoss.html#crossentropyloss
            logit_T: Float[torch.Tensor, "n_batch n_vocab n_tokens"] = logit.transpose(
                -2, -1
            )

            loss = torch.nn.functional.cross_entropy(logit_T,target)
        else:
            loss = None

        return logit, loss
    
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def init_weights(self):
        self.apply(self._init_weights)