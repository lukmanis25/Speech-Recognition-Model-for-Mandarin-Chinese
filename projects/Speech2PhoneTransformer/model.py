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
                                                dim_feedforward=n_hidden, num_encoder_layers=n_layers, num_decoder_layers=n_layers)

        # self.custom_encoder = PositionalEncoding(ninp, dropout)

        self.n_inputs = n_inputs

        # TODO: Embedding eats indices (int)!!
        # w whisperze są klasy:
        #   AudioEncoder
        #   TextDecoder (tu Transormer, nn.Embedding)
        # 
        # https://huggingface.co/learn/audio-course/en/chapter3/introduction
        # wav input -> cnn encoder -> nn.TransformerEncoder
        # Text output
        # (nn.TransformerDecoder ?)
        # The goal of an automatic speech recognition model is to predict a sequence of text tokens. This is done by adding a language modeling head — typically a single linear layer — followed by a softmax on top of the transformer’s output. This predicts the probabilities over the text tokens in the vocabulary.

        # in general the encoder outputs Embeddings
        # or do I put the whole phone_set as an embedding??

        # https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/wav2vec/wav2vec2_asr.py
        # FAIRSEQ WAV2VEC2

        self.embedding = nn.Embedding(n_tokens, n_inputs)
        self.decoder_lin = nn.Linear(n_inputs, n_tokens) # TransformerDecoder
        
        # self.decoder - TransformerDecoder is a stack of N decoder layers.
        

        self.init_emb()
    
    def init_emb(self):
        init_range = 0.1
        nn.init.uniform_(self.embedding.weight, -init_range, init_range)
        # nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder_lin.weight, -init_range, init_range)
        # super(PinyinTransformer, self).init_weights()

    def forward(self, src, tgt, has_mask=True):
        """
        src - (B, )
        mask - device magic
        """
        # src = self.embedding(src) * math.sqrt(self.n_inputs)
        # TODO: embedding -> positional encoding

        self.encoder: nn.TransformerEncoder
        self.decoder: nn.TransformerDecoder
        # src = self.positional_encoding(src)
        memory = self.encoder(src, mask=None)
        target_emb = self.embedding(tgt)
        output = self.decoder(target_emb, memory)
        output = self.decoder_lin(output)
        return F.log_softmax(output, dim=-1)

    def recognize(self, src, phones):
        """
        https://github.com/foamliu/Speech-Transformer/blob/master/transformer/transformer.py#L38
        """
        memory = self.encoder(src, mask=None)
        output = self.decoder(phones, memory)
        output = self.decoder_lin(output)
        return output