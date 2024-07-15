from .block import *

class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.pretreatment = nn.ModuleDict(dict(
            wpe=nn.Embedding(config.block_size, config.n_embd) if config.pe else PositionEmbedding(config),
            drop=nn.Dropout(config.dropout),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.enc_layer)])

    def forward(self, input, mask=None):
        device = input.device
        b, t, _ = input.size()

        pos = torch.arange(0, t, dtype=torch.long, device=device)
        pos_emb = self.pretreatment.wpe(pos)
        input = self.pretreatment.drop(input + pos_emb)

        for layer in self.layers:
            input = layer(input, mask=mask)

        return self.pretreatment.ln_f(input)


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.pretreatment = nn.ModuleDict(dict(
            wte=WordEmbedding(config),
            wpe=nn.Embedding(config.block_size, config.n_embd) if config.pe else PositionEmbedding(config),
            drop=nn.Dropout(config.dropout),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.dec_layer)])

    def forward(self, enc, t_idx, causal_mask=None, padding_mask=None):
        device = t_idx.device
        t_b, t_t = t_idx.size()

        target = self.pretreatment.wte(t_idx)
        pos = torch.arange(0, t_t, dtype=torch.long, device=device)
        target = self.pretreatment.drop(target + self.pretreatment.wpe(pos))

        for layer in self.layers:
            target = layer(enc, target, causal_mask, padding_mask)

        return self.pretreatment.ln_f(target)


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

        self.apply(self._init_weights)
        n_layers = (config.enc_layer + config.dec_layer)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_normal_(module.weight)

    def inference(self):
        pass

    def forward(self, input, target, padding_mask=None):
        _, T = target.size()
        enc = self.encoder(input, mask=padding_mask)
        out = self.decoder(enc, target, causal_mask=self.mask[:, :, :T, :T], padding_mask=padding_mask)
        logits = self.lm_head(out)
        # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=0)
        return logits