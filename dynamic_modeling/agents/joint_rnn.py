import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from tasks import JointPredTask
from agents.utils import RandomEncoding

from torch.nn.utils.rnn import pad_packed_sequence as unpack

# I guess we can have a version without BPTT
# We can write that into the state_mlp.py
# because we wouldn't need hiddens

class BaseTorchSeqAgent(JointPredTask):
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
        X = batch['feats']
        state = batch['state'].long()
        action = batch['action'].long()
        seq_len = batch['seq_len'].long()
        pre_test_scores = batch['pre_test_scores'].float()
        state_output, answer_output, scores, hidden = self.model(X, state, action, seq_len, pre_test_scores, hiddens=hiddens)
        return state_output.squeeze(2), answer_output.squeeze(2), scores, hidden

    def get_answer_loss(self, logits, batch):
        y = batch['answers'].long()
        batch_size = logits.size(0)
        seq_len = logits.size(1)

        # mask = batch['masks'].long()  # batch_size x seq_len
        mask = batch['action_masks'].long()
        device = logits.device

        logits_flat = logits.reshape(batch_size * seq_len, -1)
        y_flat = y.view(batch_size * seq_len, -1)
        mask_flat = mask.view(-1)

        with torch.no_grad():
            indices = mask_flat.cpu().numpy()
            indices = np.where(indices == 1)[0]
            indices = torch.LongTensor(indices).to(device)

        logits_mask = torch.index_select(logits_flat, 0, indices)
        y_mask = torch.index_select(y_flat, 0, indices)

        loss = F.cross_entropy(logits_mask, y_mask.view(-1).long())

        return loss, logits_mask, y_mask

    def get_state_loss(self, logits, batch):
        y = batch['targets'].long()
        batch_size = logits.size(0)
        seq_len = logits.size(1)
        mask = batch['masks'].long()  # batch_size x seq_len
        device = logits.device

        logits_flat = logits.reshape(batch_size * seq_len, -1)
        y_flat = y.view(batch_size * seq_len, -1)
        mask_flat = mask.view(-1)

        with torch.no_grad():
            indices = mask_flat.cpu().numpy()
            indices = np.where(indices == 1)[0]
            indices = torch.LongTensor(indices).to(device)

        logits_mask = torch.index_select(logits_flat, 0, indices)
        y_mask = torch.index_select(y_flat, 0, indices)

        loss = F.cross_entropy(logits_mask, y_mask.view(-1).long())

        return loss, logits_mask, y_mask

    def get_score_loss(self, logits, batch):
        y = batch['score_targets']

        # can try other errors
        # loss = torch.mean((logits - y) ** 2)
        loss = torch.mean(torch.abs(logits - y))

        return loss, logits, y

    def get_loss(self, batch, hiddens=None):
        state_output, answer_output, scores, hidden = self(batch, hiddens)
        state_loss, state_pred_prob, state_y = self.get_state_loss(state_output, batch)
        answer_loss, answer_pred_prob, answer_y = self.get_answer_loss(answer_output, batch)
        score_loss, score_pred, score_y = self.get_score_loss(scores, batch)

        # can add a weight between them
        # total_loss = state_loss + answer_loss + score_loss

        return (state_loss, state_pred_prob, state_y), (answer_loss, answer_pred_prob, answer_y), \
                (score_loss, score_pred, score_y)

    # Truncated back-propagation through time
    def training_step(self, batch, batch_idx, hiddens=None):
        tups = self.get_loss(batch, hiddens)

        # can add a weight between them
        # scaling up is required for score_loss (way too small)
        total_loss = tups[0][0] + tups[1][0] + tups[2][0] * 2 # * 3

        self.logger.experiment.log({'total_loss': total_loss,
                                    'state_loss': tups[0][0],
                                    'answer_loss': tups[1][0],
                                    'score_loss': tups[2][0],
                                    'batch': self.global_step})
        return dict(loss=total_loss)

    def validation_step(self, batch, batch_idx):
        tups = self.get_loss(batch)
        total_loss = tups[0][0] + tups[1][0] + tups[2][0]
        # state
        pred_p = F.softmax(tups[0][1], dim=1)
        self.log_preds(pred_p, tups[0][2], 'state')

        # answer
        pred_p = F.softmax(tups[1][1], dim=1)
        self.log_preds(pred_p, tups[1][2], 'answer')

        # score
        self.log_preds(tups[2][1], tups[2][2], 'score')

        log = {'val_loss': total_loss}
        self.logger.experiment.log(log)

        return {'val_loss': total_loss, 'epoch': self.current_epoch}

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
            config.dataset.num_states,
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

        self.state_decoder = nn.Linear(hidden_dim, output_dim)  # this goes to number of states

        self.answer_decoder = nn.Linear(hidden_dim, 3)

        self.score_decoder = nn.Sequential(
            nn.Linear(hidden_dim + 8, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, 8)
        )

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

        # map it to state label
        state_output = self.state_decoder(output)

        # map it to answer label
        answer_output = self.answer_decoder(output)

        # post-test score decoder
        final_feats = torch.cat([hidden[0].squeeze(0), pre_test_scores], dim=1)
        scores = self.score_decoder(final_feats) # hidden state (num_layers, batch_size, hidden state)

        return state_output, answer_output, scores, hidden