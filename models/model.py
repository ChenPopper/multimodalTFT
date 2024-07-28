from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed

from layers import (
    FeedForward,
    ResnetBlock,
    FeaturePyramid,
    CrossAttention,
    SelfAttention,
    RecurrentAttention
)


class TropicalCyclone(nn.Module):
    """
    A predictor for the intensity of tropical cyclone with multi-modal inputs,
    e.g., era5, historical intensity series, etc.

    """

    def __init__(self, configs, device=torch.device('cpu')):
        super().__init__()
        # building the feature embedding network for extracting the features from the era5 data
        conv_layers = [
            nn.Conv2d(**configs['model']['feature_pyramid']['conv_1']),
            nn.Conv2d(**configs['model']['feature_pyramid']['conv_2']),
            nn.Conv2d(**configs['model']['feature_pyramid']['conv_3'])
        ]
        resnet_layers = [
            ResnetBlock(**configs['model']['feature_pyramid']['resnet_1']),
            ResnetBlock(**configs['model']['feature_pyramid']['resnet_2']),
            ResnetBlock(**configs['model']['feature_pyramid']['resnet_3']),
            ResnetBlock(**configs['model']['feature_pyramid']['resnet_4']),
        ]
        patching_layers = [
            PatchEmbed(**configs['model']['feature_pyramid']['patching_layer_1']).to(device),
            PatchEmbed(**configs['model']['feature_pyramid']['patching_layer_2']).to(device),
            PatchEmbed(**configs['model']['feature_pyramid']['patching_layer_3']).to(device),
            PatchEmbed(**configs['model']['feature_pyramid']['patching_layer_4']).to(device),
        ]
        upsampler = nn.Upsample(scale_factor=configs['model']['feature_pyramid']['upsampler']['scale_factor'])
        self.feature_extractor = FeaturePyramid(
            conv_layers=conv_layers,
            resnet_layers=resnet_layers,
            patching_layers=patching_layers,
            upsampler=upsampler,
            device=device
        )

        # embedding the time series into the latent space (used to build the relationship to era5 features)
        self.ts_embeder = nn.Linear(configs['data']['time_series']['channel'],
                                    configs['model']['feature_pyramid']['patching']['token']['embed_dim'])

        # fusing the latent intensities and latent era5 features
        if configs['model']['feature_fuser_type'] == 'cross-attention':
            self.feature_fuser = CrossAttention(**configs['model'][configs['model']['feature_fuser_type']])
        elif configs['model']['feature_fuser_type'] == 'self-attention':
            self.feature_fuser = SelfAttention(**configs['model'][configs['model']['feature_fuser_type']])

        self.fc = nn.Linear(
            configs['data']['time_series']['channel'],
            configs['model']['feature_pyramid']['patching']['token']['embed_dim']
        )
        self.decoder = FeedForward(
            input_dim=configs['model']['feature_pyramid']['patching']['token']['embed_dim'],
            output_dim=configs['data']['time_series']['channel'],
            hidden_dims=[16, 8]
        )
        # self.decoder = ARDecoder(**configs['decoder'])
        self.decoder = RecurrentAttention(
            target_dim=configs['data']['time_series']['channel'],
            hidden_dim=configs['model']['feature_pyramid']['patching']['token']['embed_dim'],
            prediction_length=configs['data']['prediction_length'],
            num_layers=configs['model']['decoder']['num_layers'],
            attn=CrossAttention(**configs['model']['decoder']['cross-attention']),
            dropout_rate=configs['model']['decoder']['dropout_rate']
        )

    def forward(self, x: torch.Tensor, features: List[torch.Tensor]):
        """
        x: shape -> (B, T, C_x)
        features: shape -> (B, T, C_f, H, W) and
        """
        b, t, c, h, w = features[0].size()
        feat_k = []

        for i in range(t):
            _feats = [feat[:, i, ...] for feat in features]
            feats = self.feature_extractor(*_feats)  # shape -> (B, C, E)
            feat_k.append(feats)

        x_emb = self.ts_embeder(x).unsqueeze(2)  # shape -> (B, T, 1, E)
        _, c, e = feats.size()
        # feats = feats.view(-1, c, e)
        # x_emb = x_emb.view(-1, 1, e)
        feats_emb = torch.stack(feat_k, dim=1)

        h_state = self.feature_fuser(x_emb, feats_emb)  # (B, T*(T+1)/2, E)
        # x_emb_2 = self.fc(x[:, -1:, :])  # .view(b, 1, e)  # (B, 1, E)
        # hx = h_state + x_emb_2
        # output = self.decoder(hx).squeeze(2)  # (B*T, 1, C_x)
        output = self.decoder(x[:, -1:, :], h_state)[0].squeeze(-2)

        return output


class TCCorrrectionMMTFT(nn.Module):
    """
    A predictor for the intensity of tropical cyclone with multi-modal inputs,
    e.g., era5, historical intensity series, etc.

    """

    def __init__(self, configs, device=torch.device('cpu')):
        super().__init__()
        self.sample_length = configs['data']['sample_length']
        self.target_periods = configs['data']['target_periods']
        self.feature_fuser_type = configs['model']['feature_fuser_type']
        self.num_lstm_layers = configs['model']['tft']['lstm']['num_layers']
        self.device = device
        # building the feature embedding network for extracting the features from the era5 data
        self.se_layer = CBAM(channel=configs['data']['input_1']['channel'], reduction=4)

        conv_layers = [
            nn.Conv2d(**configs['model']['feature_pyramid']['conv_1']),
            nn.Conv2d(**configs['model']['feature_pyramid']['conv_2']),
            nn.Conv2d(**configs['model']['feature_pyramid']['conv_3']),
        ]
        resnet_layers = [
            ResnetBlock(**configs['model']['feature_pyramid']['resnet_1']),
            ResnetBlock(**configs['model']['feature_pyramid']['resnet_2']),
            ResnetBlock(**configs['model']['feature_pyramid']['resnet_3']),
            ResnetBlock(**configs['model']['feature_pyramid']['resnet_4']),
        ]
        patching_layers = [
            PatchEmbed(**configs['model']['feature_pyramid']['patching_layer_1']).to(device),
            PatchEmbed(**configs['model']['feature_pyramid']['patching_layer_2']).to(device),
            PatchEmbed(**configs['model']['feature_pyramid']['patching_layer_3']).to(device),
            PatchEmbed(**configs['model']['feature_pyramid']['patching_layer_4']).to(device),
        ]
        upsampler = nn.Upsample(scale_factor=configs['model']['feature_pyramid']['upsampler']['scale_factor'])
        self.feature_extractor = FeaturePyramid(
            conv_layers=conv_layers,
            resnet_layers=resnet_layers,
            patching_layers=patching_layers,
            upsampler=upsampler,
            device=device
        )

        # latent era5 features
        self.feature_fuser = nn.ModuleList(
            [
                SelfAttention(**configs['model']['self-attention'])
                for _ in range(self.sample_length)
            ]
        )
        self.fuser_output_layer = FeedForward(
            input_dim=configs['model']['fuser_output_layer']['input_channels'],
            output_dim=configs['model']['fuser_output_layer']['output_channels'],
            hidden_dims=[256, 128],
            dropout=configs['model']['fuser_output_layer']['dropout'],
        )

        # using tft encoder-decoder to deal with covariates of the
        self.target_proj = FeatureProjector(
            configs['data']['time_series']['channel'],
            configs['data']['time_series']['channel'] * configs['model']['tft']['d_model']
        )
        self.static_embedder = CatFeatureEmbeder(
            [configs['data']['num_tc'], 8],  # 8 indicates number of ocean areas
            configs['model']['tft']['d_model']
        )
        self.tropical_region_embedder = CatFeatureEmbeder([3], configs['model']['tft']['d_model'])
        self.real_covar_proj = FeatureProjector(
            configs['model']['tft']['real_dim'],
            configs['model']['tft']['d_model'] * configs['model']['tft']['real_dim']
        )
        # self.past_real_proj = FeatureProjector(
        #     configs['model']['tft']['past_real_dim'],
        #     configs['model']['tft']['d_model'] * configs['model']['tft']['past_real_dim']
        # )

        self.static_selector = VariableSelectionNetwork(
            d_hidden=configs['model']['tft']['d_model'],
            num_vars=configs['model']['tft']['static_dim'],
            dropout=configs['model']['tft']['dropout'],
        )
        img_out_channels = configs['model']['fuser_output_layer']['output_channels']

        # num_vars = (configs['data']['time_series']['channel']
        #             + configs['model']['tft']['dyn_cat_dim']
        #             + configs['model']['tft']['real_dim']  # + configs['model']['tft']['real_dim']  # + 1
        #             + configs['model']['tft']['past_real_dim']  # + configs['model']['tft']['past_real_dim'] # + 1
        #             + img_out_channels)

        num_vars = configs['model']['tft']['dyn_cat_dim'] + configs['model']['tft']['real_dim'] + img_out_channels

        self.ctx_selector = VariableSelectionNetwork(
            d_hidden=configs['model']['tft']['d_model'],
            num_vars=num_vars,  # +1 is the covariates of the era5 image cross-attn feats
            add_static=True,
            dropout=configs['model']['tft']['dropout'],
        )
        # self.tgt_selector = VariableSelectionNetwork(
        #     d_hidden=configs['model']['tft']['d_model'],
        #     num_vars=configs['model']['tft']['real_dim'],  # 1
        #     add_static=True,
        #     dropout=configs['model']['tft']['dropout'],
        # )
        self.selection = GatedResidualNetwork(
            d_hidden=configs['model']['tft']['d_model'],
            dropout=configs['model']['tft']['dropout'],
        )
        self.enrichment = GatedResidualNetwork(
            d_hidden=configs['model']['tft']['d_model'],
            dropout=configs['model']['tft']['dropout'],
        )
        self.state_h = GatedResidualNetwork(
            d_hidden=configs['model']['tft']['d_model'],
            d_output=configs['model']['tft']['d_model'],
            dropout=configs['model']['tft']['dropout'],
        )
        self.state_c = GatedResidualNetwork(
            d_hidden=configs['model']['tft']['d_model'],
            d_output=configs['model']['tft']['d_model'],
            dropout=configs['model']['tft']['dropout'],
        )

        self.bidirection = configs['model']['tft']['lstm']['bidirection']
        self.lstm_encoder = LSTM(input_size=configs['model']['tft']['d_model'],
                                 hidden_size=configs['model']['tft']['d_model'],
                                 num_layers=configs['model']['tft']['lstm']['num_layers'],
                                 bidirectional=self.bidirection,
                                 dropout=configs['model']['tft']['dropout'] if configs['model']['tft']['lstm'][
                                     'num_layers'] else 0,
                                 batch_first=True)

        self.lstm_post_gate = GatedLinearUnit(input_size=configs['model']['tft']['d_model'] * (1 + self.bidirection))

        self.lstm_post_norm = nn.LayerNorm(configs['model']['tft']['d_model'] * (1 + self.bidirection))

        self.static_enrich = GatedResidualNetwork(
            d_input=configs['model']['tft']['d_model'] * (1 + self.bidirection),
            d_hidden=configs['model']['tft']['d_model'],
            d_static=configs['model']['tft']['d_model'],
            d_output=configs['model']['tft']['d_model'] * (1 + self.bidirection),
            dropout=configs['model']['tft']['dropout']
        )

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=configs['model']['tft']['d_model'] * (1 + self.bidirection),
            kdim=configs['model']['tft']['d_model'] * (1 + self.bidirection),
            vdim=configs['model']['tft']['d_model'] * (1 + self.bidirection),
            num_heads=configs['model']['tft']['attn']['num_head'],
            dropout=configs['model']['tft']['dropout']
        )
        self.post_attn_gate = GatedLinearUnit(input_size=configs['model']['tft']['d_model'] * (1 + self.bidirection))
        self.post_attn_norm = nn.LayerNorm(configs['model']['tft']['d_model'] * (1 + self.bidirection))
        self.pos_wise_ff = GatedResidualNetwork(
            d_hidden=configs['model']['tft']['d_model'] * (1 + self.bidirection),
            dropout=configs['model']['tft']['dropout'] * (1 + self.bidirection),
        )

        self.pre_output_gate = GatedLinearUnit(input_size=configs['model']['tft']['d_model'] * (1 + self.bidirection))
        self.pre_output_norm = nn.LayerNorm(configs['model']['tft']['d_model'] * (1 + self.bidirection))
        self.output_layer = nn.Linear(configs['model']['tft']['d_model'] * (1 + self.bidirection),
                                      configs['data']['time_series']['channel']
                                      )
        self.correction_layer = FeedForward(
            configs['data']['sample_length'],
            configs['data']['target_periods'],
            hidden_dims=[32, 8]
        ) if configs['data']['sample_length'] != configs['data']['target_periods'] else None

    def forward(
            self,
            img_features: List[torch.Tensor],
            covariates: torch.Tensor
    ):
        """
        img_features: shape -> (B, T, C_f, H, W) and
        covariates: (B, hist_len, C_pc)
        """
        (
            covariates,
            static_covariates,
            feat_names
        ) = self._preprocess(
            feat_static_cat=covariates[:, 0, 0:2].to(torch.long),
            feat_dynamic_real=covariates[:, :, 2:8],
            feat_dynamic_cat=covariates[..., -1:].to(torch.long)
        )
        static_var, static_weights = self.static_selector(static_covariates)  # [B, d_model]
        c_selection = self.selection(static_var).unsqueeze(1)  # [B, 1, d_model]
        c_enrichment = torch.repeat_interleave(
            self.enrichment(static_var).unsqueeze(1),
            self.sample_length,
            dim=1
        )
        c_h = self.state_h(static_var)  # [B, d_model]
        c_c = self.state_c(static_var)  # [B, d_model]
        states = [c_h.unsqueeze(0), c_c.unsqueeze(0)]

        b, t, c, h, w = img_features[0].size()
        feat_k = []
        channel_weights = []
        for i in range(t):
            _feats, _weights = self.se_layer(img_features[0][:, i, ...])
            # channel_weights.append(_weights)
            feats = self.feature_extractor(_feats)  # shape -> (B, C, E)
            feat_k.append(feats)
        # channel_weights = torch.stack(channel_weights, dim=1)  # (B, T, C)

        _, c, e = feats.size()

        self_attn = [
            self.fuser_output_layer(fuser(feat).transpose(-2, -1)).transpose(-2, -1)  # (B, C_in, E)
            for fuser, feat in zip(self.feature_fuser, feat_k)
        ]  # [(B, C_in, E), ...]
        img_covs = torch.stack(self_attn, dim=1)
        covariates.extend([
            img_covs[..., c, :] for c in range(img_covs.shape[-2])
        ])
        feat_names.extend([f'img_feat_{i}' for i in range(img_covs.shape[-2])])

        # states = [torch.repeat_interleave(c_h.unsqueeze(0), self.num_lstm_layers, dim=0),
        #           torch.repeat_interleave(c_c.unsqueeze(0), self.num_lstm_layers, dim=0)]
        ctx_input, ctx_weights = self.ctx_selector(
            covariates, c_selection
        )  # [B, hist_len, d_model]
        # tgt_input, tgt_weights = self.tgt_selector(
        #     future_covariates, c_selection
        # )  # [B, pred_len, d_model]

        encoding = self.ts_encoder(
            ctx_input, states
        )  # [B, hist_len + pred_len, d_model]
        lstm_output = self.lstm_post_gate(encoding)
        lstm_output = self.lstm_post_norm(lstm_output)
        attn_input = self.static_enrich(lstm_output, c_enrichment)
        decoding, attn_weights = self.mask_attn_decoder(
            attn_input,
        )  # [B, pred_len, d_hidden]
        attn_output = self.post_attn_gate(decoding) + attn_input  # [:, self.history_length:, ...]
        attn_output = self.post_attn_norm(attn_output)
        output = self.pos_wise_ff(attn_output)  # [self.history_length:,:,:])

        ##skip connection over Decoder
        output = self.pre_output_gate(output) + lstm_output  # [:, self.history_length:, :]

        # Final output layers
        output = self.pre_output_norm(output)
        output = self.output_layer(output)

        if self.correction_layer is not None:
            output = self.correction_layer(output.transpose(-2, -1)).transpose(-2, -1)  # (B, target_periods, C)

        feature_weights = self.get_weights(static_weights, ctx_weights, channel_weights, attn_weights)
        feature_weights.update({
            'feat_names': feat_names
        })

        return output, feature_weights

    def ts_encoder(self, ctx_input, states: Optional[List[torch.Tensor]]):
        ctx_encodings, h_state = self.lstm_encoder(ctx_input, states)
        encodings = ctx_encodings
        skip = ctx_input.repeat_interleave(1 + self.lstm_encoder.bidirectional, dim=-1)
        output = self.lstm_post_gate(encodings + skip)
        output = self.lstm_post_norm(output)
        return output

    def mask_attn_decoder(self, x):
        # x: (B, hist_len + pred_len, E)
        x = x.transpose(0, 1)
        attn_output, attn_weights = self.multihead_attn(x, x, x)
        output = self.post_attn_gate(attn_output) + x
        output = self.post_attn_norm(output).transpose(0, 1)
        return output, attn_weights

    def _preprocess(
            self,
            # feat_static_real: Optional[torch.Tensor],  # [B, D_sr]
            feat_static_cat: Optional[torch.Tensor],  # [B, D_sc]  tc name number
            feat_dynamic_real: Optional[torch.Tensor],  # [B, T + H, D_dr]
            feat_dynamic_cat: Optional[torch.Tensor],  # [B, T + H, D_dc]
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[str]
    ]:
        covariates = []
        feat_names = []
        static_covariates = []

        embs = self.tropical_region_embedder(feat_dynamic_cat)
        covariates.extend(embs)
        feat_names.extend(['feat_dynamic_cat'] * len(embs))
        fdr_num_chunks = feat_dynamic_real.shape[-1]  # 1
        proj = list(torch.chunk(self.real_covar_proj(feat_dynamic_real), chunks=fdr_num_chunks, dim=-1))
        covariates.extend(proj)
        feat_names.extend(['feat_dynamic_real'] * fdr_num_chunks)
        # future_covariates.extend([ele[:, self.history_length:, ...] for ele in proj])
        # future_feat_names.extend(['feat_dynamic_real'] * feat_dynamic_real.shape[-1])
        # pfdr_num_chunks = past_feat_dynamic_real.shape[-1]  # 1
        # past_covariates.extend(
        #     list(torch.chunk(self.past_real_proj(past_feat_dynamic_real), chunks=pfdr_num_chunks, dim=-1))
        # )
        # past_feat_names.extend(['past_feat_dynamic_real'] * pfdr_num_chunks)

        embs = self.static_embedder(feat_static_cat)  # [(, d_model)]
        static_covariates.extend(embs)

        return (
            covariates,
            static_covariates,
            feat_names
        )

    @staticmethod
    def get_weights(static_weights, ctx_weights, channel_weights, attn_weights):
        sw = torch.mean(static_weights, dim=0).detach()  # (C_s,)
        cw = torch.mean(ctx_weights, dim=0).detach()  # (hist_len, C_p)
        # chw = torch.mean(channel_weights, dim=0).detach()
        # tw = torch.mean(tgt_weights, dim=0).detach()  # (pred_len, C_f)
        aw = torch.mean(attn_weights, dim=0).detach()  # (hist_len, pred_len)
        feature_weights = dict(
            static_weights=sw,
            feature_weights=cw,
            # channel_weights=chw,
            # future_feature_weights=tw,
            attn_feature_weights=aw
        )
        return feature_weights
