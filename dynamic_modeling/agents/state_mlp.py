import torch
import torch.nn as nn
import torch.nn.functional as F

from tasks import StatePredTask
from agents.utils import RandomEncoding


class BaseTorchAgent(StatePredTask):
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

    def forward(self, batch):
        X = batch['feats']
        state = batch['state'].long()
        action = batch['action'].long()
        out = self.model(X, state, action)
        return out

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        y = batch['targets']

        #loss = F.binary_cross_entropy_with_logits(logits, y.float())
        loss = F.cross_entropy(logits, y.long()) # logtis: (N, C); y: (N,)

        self.logger.experiment.log({'train_loss': loss, 'batch': self.global_step})

        return dict(loss=loss)

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        y = batch['targets']
        # loss = F.binary_cross_entropy_with_logits(logits, y.float())
        loss = F.cross_entropy(logits, y.long())

        pred_p = F.softmax(logits, dim=1)  # torch.argmax(logits, dim=1)
        self.log_preds(pred_p, y)

        self.logger.experiment.log({'val_loss': loss})

        return {'val_loss': loss, 'epoch': self.current_epoch}

    def configure_optimizers(self):
        optim_config = self.config.optimizer
        return torch.optim.Adam(
            self.parameters(),
            lr=optim_config.learning_rate,
            weight_decay=optim_config.weight_decay)


class MLPAgent(BaseTorchAgent):

    def create_model(self, config):
        model = EdNetMLP(
            config.dataset.feat_dim,
            config.dataset.num_states,
            config.model.hidden_dim,
            config.model.encoding_dim,
            config.model.encoding_type,
            # times 2 to leave room for new users
            config.dataset.num_states,
            config.dataset.num_actions
        )
        return model


class EdNetMLP(nn.Module):

    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim=16,
            encoding_dim=16,
            encoding_type='random',
            num_states=480+1,
            num_actions=3
    ):
        super().__init__()
        assert encoding_type in ['random', 'learned']
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + 2 * encoding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        if encoding_type == 'random':
            self.state_enc = RandomEncoding(encoding_dim, max_len=num_states)
            self.action_enc = RandomEncoding(num_actions)
        elif encoding_type == 'learned':
            self.state_enc = nn.Embedding(num_states, encoding_dim)
            self.action_enc = nn.Embedding(num_actions, encoding_dim)
        else:
            raise Exception(f'Encoding type {encoding_type} not supported.')

    def forward(self, inputs, state, action):
        state_embs = self.state_enc(state)
        action_embs = self.action_enc(action)

        inputs = torch.cat([inputs, state_embs, action_embs], dim=1)
        logits = self.mlp(inputs)
        return logits
