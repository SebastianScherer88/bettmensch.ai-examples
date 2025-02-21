from typing import Callable, Optional, Union, Tuple, Any

import torch
from jaxtyping import Bool, Float, Int
import torch.utils


class VerboseIOModule(torch.nn.Module):
    """Utility Mixin to allow for toggling of the IO shape display at runtime.
    Useful for debugging model architecture."""

    _id: Optional[str] = None
    _verbose: bool = False
    _nest_level: Optional[int] = None

    def __init__(self, id: Optional[str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._id = id
        # self.forward = self.display_io_sizes(id)(self.forward)

    def set_io_verbosity(self, value: bool):
        self._verbose = value

        # set on all child modules
        for module in self.modules():
            if isinstance(module, VerboseIOModule) and module != self:
                module._verbose = value

    def set_nest_level(self, level: int = 0):
        assert (
            self.nest_level is None
        ), f"Nest level {self.nest_level} has already been set and "
        f"cannot be re-set for {self.id}"
        self._nest_level = level

        # cascade to child modules in nested fashion to generate correct nest
        # level we need to skip modules that have the level set already as each
        # module has line of sight of ALL its child modules' descendants (as
        # opposed to just its own child modules)
        for module in self.modules():
            if (
                isinstance(module, VerboseIOModule)
                and (module != self)  # noqa: W503
                and (module.nest_level is None)  # noqa: W503
            ):
                module.set_nest_level(level=self.nest_level + 1)

    @property
    def id(self) -> Union[str, None]:
        return self._id

    @property
    def is_verbose(self) -> bool:
        return self._verbose

    @property
    def nest_level(self) -> int:
        return self._nest_level

    @staticmethod
    def display_io_sizes(id: Optional[str] = None):
        def wrapper(forward: Callable):
            def func(*args, **kwargs):

                parent_module_mixin = args[0]
                assert isinstance(parent_module_mixin, VerboseIOModule)
                if id is not None:
                    display_id = id
                elif parent_module_mixin.id is not None:
                    display_id = parent_module_mixin.id
                else:
                    display_id = str(type(parent_module_mixin))

                indent = parent_module_mixin.nest_level * "    "
                verbose = parent_module_mixin.is_verbose

                if verbose:
                    for i, arg in enumerate(args[1:]):
                        try:
                            print(
                                f"{indent}[{display_id}]'s positional input"
                                f" {i+1}'s size: {arg.size()}"
                            )
                            print(
                                f"{indent}[{display_id}]'s positional input"
                                f" {i+1}'s GPU device: {arg.get_device()}"
                            )
                        except AttributeError:
                            print(
                                f"{indent}[{display_id}]'s positional input"
                                f" {i+1} is not a torch tensor: {type(arg)}"
                            )

                    for arg_name, arg_value in kwargs.items():
                        try:
                            print(
                                f"{indent}[{display_id}]'s named input"
                                f" {arg_name}'s size: {arg_value.size()}"
                            )
                            print(
                                f"{indent}[{display_id}]'s name input"
                                f" {arg_name}'s GPU device: "
                                f"{arg_value.get_device()}"
                            )
                        except AttributeError:
                            continue

                result = forward(*args, **kwargs)

                if verbose:
                    try:
                        print(
                            f"{indent}[{display_id}]'s output's size:"
                            f" {result.size()}"
                        )
                    except AttributeError:
                        print(
                            f"{indent}[{display_id}]'s output is not a torch"
                            f" tensor: {type(result)}"
                        )

                return result

            return func

        return wrapper


class Embedding(VerboseIOModule):

    positions: Int[torch.nn.UninitializedBuffer, "-1"]

    def __init__(
        self,
        n_vocab: int,
        n_tokens: int = 512,
        dim_embed: int = 768,
        dropout: float = 0.1,
        id: str = "Embedding",
    ):
        super().__init__(id=id)
        self.max_length = n_tokens
        self.token = torch.nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=dim_embed,
            dtype=torch.half
        )
        self.pos = torch.nn.Embedding(
            num_embeddings=n_tokens,
            embedding_dim=dim_embed,
            dtype=torch.half
        )
        self.dropout = torch.nn.Dropout(p=dropout)
        self.register_buffer("positions", positions = torch.range(0,self.max_length,dtype=torch.half))

    @VerboseIOModule.display_io_sizes()
    def forward(
        self, x: Float[torch.Tensor, "n_batch n_tokens"]
    ) -> Float[torch.Tensor, "n_batch n_tokens dim_embed"]:

        e_token: Float[
            torch.Tensor, "n_batch n_tokens dim_embed"
        ] = self.token(x)

        e_pos: Float[torch.tensor, "1 n_tokens dim_embed"] = self.pos(self.positions).unsqueeze(0)

        e = self.dropout(e_token + e_pos)

        return e


def generate_padded_subsequent_mask(
    input_mask: Bool[torch.Tensor, "n_batch n_tokens"]
) -> Float[torch.Tensor, "n_batch n_tokens n_tokens"]:
    """Generates a mask that, when passed to the attention function, forces the
      coefficients to be 0 for
    - values/tokens that come after the query/token in question in the original
        input sequence
    - values/tokens that stem from padded tokens in the original input
        sequence,
    thus forcing the attention to the non-trivial set of values/tokens. This
    reduces noise during optimization, which speeds up training and improves
     model performance
    """

    device = input_mask.get_device()

    n_tokens = input_mask.size()[-1]
    padded_token_mask: Float[
        torch.Tensor, "n_batch 1 n_tokens"
    ] = input_mask.unsqueeze(1)
    subsequent_token_mask: Float[torch.Tensor, "1 n_tokens n_tokens"] = (
        torch.tril(torch.ones(size=(n_tokens, n_tokens)))
        .type(torch.bool)
        .unsqueeze(0)
    )

    if device != -1:
        subsequent_token_mask = subsequent_token_mask.to(device)

    final_mask = (padded_token_mask & subsequent_token_mask)

    if device != -1:
        final_mask = final_mask.to(device)

    return final_mask


class MultiHeadedSelfAttention(VerboseIOModule):
    """Implements tensorized attention based on Andrej Karpaty's MinGPT"""

    def __init__(
        self,
            n_heads: int = 12,
            dim_input: int = 768,
            dropout: float = 0.1,
            id: str = "MultiHeadAttentionKarpaty",
    ):
        super().__init__(id=id)

        assert (
            dim_input % n_heads == 0
        ), f"Query & value embedding size {dim_input} is not divisable by "
        f"attention head count {n_heads}"

        self.n_heads = n_heads
        self.dim_input = dim_input
        self.dim_sh_embed = int(dim_input / n_heads)
        self.scale = torch.sqrt(torch.tensor(self.dim_sh_embed,requires_grad=False))

        self.projection_attention = torch.nn.Linear(self.dim_input, self.dim_input * 3,dtype=torch.half)
        self.dropout_attention = torch.nn.Dropout(p=dropout)
        self.projection_out = torch.nn.Linear(self.dim_input,self.dim_input,dtype=torch.half)
        self.dropout_residual = torch.nn.Dropout(p=dropout)

    def forward(
            self,
            x: Float[torch.Tensor, "n_batch n_tokens dim_embed"],
            mask: Optional[Float[torch.Tensor, "n_batch n_tokens n_tokens"]],
        ) -> Float[torch.Tensor, "n_batch n_tokens dim_embed"]:

        n_batch, n_tokens, _ = x.size()

        # do all query, key and value projections for all heads in one go 
        # yielding 3 tensors of dimension 
        # (n_batch, n_tokens, dim_embed)
        q, k, v = self.projection_attention(x).split(self.dim_input,dim=-1)

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
        mask_h = mask.unsqueeze(1)
        coef_masked = coef.masked_fill(mask_h==False,float('-inf'))

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
        att = att_h.transpose(1,2).contiguous().view(n_batch, n_tokens, self.dim_input)

        att_proj = self.projection_out(att)

        return self.dropout_residual(att_proj)

class FeedForward(VerboseIOModule):
    """The feed forward layer (including dropout) as described in https://...
    ...cdn.openai.com/research-covers/language-unsupervised/...
    ...language_understanding_paper.pdf. Defaults to the exact configuration
     presented in the paper."""

    def __init__(
        self,
        dim_input: int = 768,
        dim_ff: int = 3072,
        dropout: float = 0.1,
        activation_class=torch.nn.GELU,
        id: str = "FeedForward",
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

        super().__init__(id=id)

        self.dim_input = dim_input
        self.dim_ff = dim_ff

        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(dim_input, dim_ff,dtype=torch.half),
            activation_class(),
            torch.nn.Linear(dim_ff, dim_input,dtype=torch.half),
            torch.nn.Dropout(p=dropout),
        )

    @VerboseIOModule.display_io_sizes()
    def forward(
        self, x: Float[torch.Tensor, "n_batch n_tokens dim_input"]
    ) -> Float[torch.Tensor, "n_batch n_tokens dim_input"]:
        return self.sequential(x)

class SkipNorm(VerboseIOModule):
    """The skip + normalization layer (including dropout) wrapper as described
    in https://cdn.openai.com/research-covers/language-unsupervised/...
    ...language_understanding_paper.pdf"""

    def __init__(
        self,
        skipped_layer: Union[MultiHeadedSelfAttention, FeedForward],
        dropout: float = 0.1,
        id: str = "SkipNorm",
    ):
        """_summary_

        Args:
            dropout (float, optional): The dropout rate. Defaults to 0.1.
        """
        super().__init__(id=id)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.skipped_layer = skipped_layer
        self.norm_layer = torch.nn.LayerNorm(
            normalized_shape=(skipped_layer.dim_input,),
            dtype=torch.half
        )

    @VerboseIOModule.display_io_sizes()
    def forward(
        self,
        x: Float[torch.Tensor, "n_batch n_tokens n_dim_input"],
        *skipped_layer_args,
        **skipped_layer_kwargs,
    ) -> Float[torch.Tensor, "n_batch n_tokens n_dim_input"]:

        return self.dropout(
            self.norm_layer(
                x
                + self.skipped_layer(  # noqa: W503
                    x, *skipped_layer_args, **skipped_layer_kwargs
                )
            )
        )

    
class DecoderLayer(VerboseIOModule):

    def __init__(
        self,
        n_heads: int = 12,
        dim_input: int = 768,
        dropout: float = 0.1,
        id: str = "DecoderLayer",
    ):
        super().__init__(id=id)
        self.skip_norm_mha = SkipNorm(
            MultiHeadedSelfAttention(
                n_heads=n_heads, dim_input=dim_input, dropout=dropout, id="mhsa_k"
            ),
            id="skip_mhsa_k",
        )
        self.skip_norm_ff = SkipNorm(
            FeedForward(dim_input=dim_input, dropout=dropout, id="ff"),
            id="skip_ff",
        )

    @VerboseIOModule.display_io_sizes()
    def forward(
        self,
        x: Float[torch.Tensor, "n_batch n_tokens dim_input"],
        mask: Bool[torch.Tensor, "n_batch n_tokens"],
    ) -> Float[torch.Tensor, "n_batch n_tokens dim_input"]:
        mha_mask = generate_padded_subsequent_mask(mask)
        mha_out = self.skip_norm_mha(x=x, mask=mha_mask)
        ff_out = self.skip_norm_ff(mha_out)

        return ff_out
        
class Decoder(VerboseIOModule):

    def __init__(
        self,
        dim_input: int = 768,
        n_decoder_layers: int = 12,
        n_heads: int = 12,
        dropout: float = 0.1,
        id: str = "Decoder",
    ):
        super().__init__(id=id)

        self.layers = torch.nn.ModuleList(
            [
                DecoderLayer(
                    n_heads=n_heads,
                    dim_input=dim_input,
                    dropout=dropout,
                    id=f"DecoderLayer_{i+1}",
                )
                for i in range(n_decoder_layers)
            ]
        )

    @VerboseIOModule.display_io_sizes()
    def forward(
        self,
        x: Float[torch.Tensor, "n_batch n_tokens dim_input"],
        mask: Bool[torch.Tensor, "n_batch n_tokens"],
    ) -> Float[torch.Tensor, "n_batch n_tokens dim_input"]:
        for layer in self.layers:
            x = layer(x, mask)

        return x
        
class GPT1Core(VerboseIOModule):

    def __init__(
        self,
        n_vocab: int,
        n_tokens=512,
        dim_embed: int = 768,
        n_decoder_layers: int = 12,
        n_heads: int = 12,
        dropout: float = 0.1,
        id: str = "GPT1Core",
    ):
        super().__init__(id=id)
        self.embedding = Embedding(
            n_vocab=n_vocab,
            n_tokens=n_tokens,
            dim_embed=dim_embed,
            dropout=dropout,
        )
        self.decoder = Decoder(
            dim_input=dim_embed,
            n_decoder_layers=n_decoder_layers,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.set_nest_level()

    @VerboseIOModule.display_io_sizes()
    def forward(
        self,
        x: Float[torch.Tensor, "n_batch n_tokens"],
        mask: Bool[torch.Tensor, "n_batch n_tokens"],
    ) -> Float[torch.Tensor, "n_batch n_tokens dim_input"]:
        e = self.embedding(x)
        d = self.decoder(e, mask)

        return d
    
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


class GPT1Pretrain(GPT1Core):
    def __init__(
        self,
        n_vocab: int,
        n_tokens=512,
        dim_embed: int = 768,
        n_decoder_layers: int = 12,
        n_heads: int = 12,
        dropout: float = 0.1,
        id: str = "GPT1Pretrain",
    ):
        super().__init__(
            n_vocab=n_vocab,
            n_tokens=n_tokens,
            dim_embed=dim_embed,
            n_decoder_layers=n_decoder_layers,
            n_heads=n_heads,
            dropout=dropout,
            id=id,
        )

        self.lm_head = torch.nn.Linear(dim_embed, n_vocab, dtype=torch.half)

    @VerboseIOModule.display_io_sizes()
    def forward(
        self,
        x: Float[torch.Tensor, "n_batch n_tokens dim_input"],
        mask: Bool[torch.Tensor, "n_batch n_tokens"],
        target: Float[torch.Tensor, "n_batch n_tokens"],
    ) -> Union[Float[torch.Tensor, "n_batch n_tokens n_vocab"],Tuple[Float[torch.Tensor, "n_batch n_tokens n_vocab"],Float[torch.Tensor, "1"]]]:
        """We extend the core class' forward method by 
        - a language head that maps the last transformer layer's outputs back
            into the vocabulary space, and
        - an optional loss that is only invoked if the `target` arg is provided
        """
        d: Float[torch.Tensor, "n_batch n_tokens dim_input"] = super().forward(x, mask)

        logit: Float[torch.Tensor, "n_batch n_tokens n_vocab"] = self.lm_head(d)

        if target is not None:
            # torch's cross entropy loss requires the vocabulary dimension in the
            # second rank as per https://pytorch.org/docs/stable/generated/...
            # ...torch.nn.CrossEntropyLoss.html#crossentropyloss
            logit_T: Float[torch.Tensor, "n_batch n_vocab n_tokens"] = logit.transpose(
                -2, -1
            )

            loss = torch.nn.functional.cross_entropy(logit_T,target)

            return logit, loss

        return logit
