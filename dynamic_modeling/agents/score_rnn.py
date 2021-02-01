"""
Very similar to answer, but with a different task

This is also NOT joint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sklearn.metrics as skm

from dynamic_modeling.tasks import ScorePredTask
from dynamic_modeling.agents.utils import RandomEncoding

# from tasks import ScorePredTask
# from agents.utils import RandomEncoding

from torch.nn.utils.rnn import pad_packed_sequence as unpack

# I guess we can have a version without BPTT
# We can write that into the state_mlp.py
# because we wouldn't need hiddens

class BaseTorchSeqAgent(ScorePredTask):
    """
    Base class for all feedforward neural network
    architectures. Note that the create_model() fn
    is not defined.
    """

    def __init__(self, config):
        super().__init__(config)

        self.model = self.create_model(config)

    def create_model(self, config):
        raise NotImplementedError

    def forward(self, batch, hiddens=None):
        # answer is not passed in, but folded as cumulative stats
        X = batch['feats']
        state = batch['state'].long()
        action = batch['action'].long()
        seq_len = batch['seq_len'].long()
        pre_test_scores = batch['pre_test_scores'].float()
        out, state = self.model(X, state, action, seq_len, pre_test_scores, hiddens=hiddens)
        return out, state

    def get_loss(self, batch, hiddens=None):
        logits, state = self(batch, hiddens)
        y = batch['score_targets']

        # can try other errors
        # loss = torch.mean((logits - y) ** 2)  # MSE is better for NLG_SCORE MAE, but produces no diff in Post-test-score
        loss = torch.mean(torch.abs(logits - y)) # MAE is better for ranking error for NLG_SCORE lol

        return loss, logits, y

    # Truncated back-propagation through time
    def training_step(self, batch, batch_idx, hiddens=None):
        loss, _, _ = self.get_loss(batch, hiddens)
        self.logger.experiment.log({'train_loss': loss, 'batch': self.global_step})
        return dict(loss=loss)

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.get_loss(batch)
        self.log_preds(logits, y)

        return {'val_loss': loss, 'epoch': self.current_epoch}

    def configure_optimizers(self):
        optim_config = self.config.optimizer
        return torch.optim.Adam(
            self.parameters(),
            lr=optim_config.learning_rate,
            weight_decay=optim_config.weight_decay)


class RNNAgent(BaseTorchSeqAgent):

    def create_model(self, config):
        model = TutorBaseRNN(
            config.dataset.feat_dim,
            8,
            'lstm',
            config.model.hidden_dim,
            config.model.encoding_dim,
            config.model.encoding_type,
            1,
            0,
            # times 2 to leave room for new users
            config.dataset.num_states,
            config.dataset.num_actions
        )
        return model


class TutorBaseRNN(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            rnn_type='lstm',
            hidden_dim=32,
            encoding_dim=16,
            encoding_type='random',
            num_layers=1,
            dropout=0,
            num_states=480 + 1,
            num_actions=3
    ):
        super().__init__()
        assert encoding_type in ['random', 'learned']
        if rnn_type == 'lstm':
            rnn_model = nn.LSTM
        elif rnn_type == 'gru':
            rnn_model = nn.GRU
        else:
            raise Exception("{} type is not a valid option".format(rnn_type))

        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = rnn_model(input_dim + 2 * encoding_dim,
                                 hidden_dim,
                                 num_layers,
                                 dropout=dropout,
                                 batch_first=True)

        if encoding_type == 'random':
            self.state_enc = RandomEncoding(encoding_dim, max_len=num_states)
            self.action_enc = RandomEncoding(num_actions)
        elif encoding_type == 'learned':
            self.state_enc = nn.Embedding(num_states, encoding_dim)
            self.action_enc = nn.Embedding(num_actions, encoding_dim)
        else:
            raise Exception(f'Encoding type {encoding_type} not supported.')

        # so best post-test score prediction:
        # 1. 0.141: without pre-test score
        # 2. 0.139: with pre-test score, complex MLP
        # 3. 0.163: with pre-test score, linear

        # self.decoder = nn.Linear(hidden_dim, output_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + 8, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, 8)
        )
        # self.decoder = nn.Linear(hidden_dim + 8, output_dim)

    def forward(self, inputs, state, action, lengths, pre_test_scores, hiddens=None):
        batch_size, max_seq_len, _ = inputs.size()

        s_embs = self.state_enc(state.view(batch_size * max_seq_len))
        s_embs = s_embs.view(batch_size, max_seq_len, -1)

        a_embs = self.action_enc(action.view(batch_size * max_seq_len))
        a_embs = a_embs.view(batch_size, max_seq_len, -1)

        inputs = torch.cat([inputs, s_embs, a_embs], dim=2)
        lengths = lengths.cpu()
        packed_emb = nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)

        if hiddens is not None:
            output, hidden = self.encoder(packed_emb, hiddens)
        else:
            output, hidden = self.encoder(packed_emb)

        output = unpack(output, padding_value=0, batch_first=True)[0]

        # map it to label
        # can add a bit of attention mix in here (note you need masking)
        final_feats = torch.cat([hidden[0].squeeze(0), pre_test_scores], dim=1)
        scores = self.decoder(final_feats)

        return scores, hidden