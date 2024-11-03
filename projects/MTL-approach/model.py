# https://github.com/pytorch/examples/blob/cdef4d43fb1a2c6c4349daa5080e4e8731c34569/word_language_model/model.py

import math
from torch import nn
import torch.nn.functional as F

class PinyinTransformer(nn.Transformer):
    def __init__(self, n_tokens, n_inputs=32, n_head=1, n_hidden=64, n_layers=2, batch_first=True):
        """
        Args
        n_tokens - size of dataset (for embeddings)

        n_inputs - aka em_size, the number of expected features in the encoder/decoder inputs (default=32 /* 512 */).

        n_head - the number of heads in the multiheadattention models (default=1 /* 8 */).

        n_layers - the number of sub-encoder-layers in the encoder (default=2 /* 6 */).

        /* num_decoder_layers - the number of sub-decoder-layers in the decoder (default=6). */

        n_hidden - the dimension of the feedforward network model (default=64/*2048*/).

        /* dropout - the dropout value (default=0.1). */

        /* activation - the activation function of encoder/decoder intermediate layer, can be a string ("relu" or "gelu") or a unary callable. Default: relu */

        /* custom_encoder - custom encoder (default=None). */

        /* custom_decoder - custom decoder (default=None). */

        /* layer_norm_eps - the eps value in layer normalization components (default=1e-5). */

        /* batch_first - If True, then the input and output tensors are provided as (batch, seq, feature). Default: False (seq, batch, feature). */

        /* norm_first - if True, encoder and decoder layers will perform LayerNorms before other attention and feedforward operations, otherwise after. Default: False (after). */

        /* bias - If set to False, Linear and LayerNorm layers will not learn an additive bias. Default: True. */
        """
        super(PinyinTransformer, self).__init__(d_model=n_inputs, nhead=n_head, batch_first=batch_first, activation="gelu",
                                                dim_feedforward=n_hidden, num_encoder_layers=n_layers)
        
        # self.custom_encoder = PositionalEncoding(ninp, dropout)

        self.n_inputs = n_inputs
        self.embedding = nn.Embedding(n_tokens, n_inputs)
        self.decoder = nn.Linear(n_inputs, n_tokens)

        self.init_weigths()
    
    def init_weigths(self):
        init_range = 0.1
        nn.init.uniform_(self.embedding.weight, -init_range, init_range)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -init_range, init_range)

    def forward(self, src, has_mask=True):
        """
        src - (B, )
        mask - device magic
        """
        src = self.embedding(src) * math.sqrt(self.n_inputs)
        # src = self.custom_encoder(src)
        output = self.encoder(src, mask=None)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)