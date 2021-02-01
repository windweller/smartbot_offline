import torch
import pytorch_lightning as pl
import wandb

import numpy as np
import sklearn.metrics as skm

# the best way to refactor this
# is create a central Task that inherits from LightningModule
# and use config to control which actual task to use

class StatePredTask(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()  # save config
        self.config = config  # config is a dotmap

        self.first_epoch = True
        self._reset_aggregates()

    def _reset_aggregates(self):
        self.pred_probs = []
        self.targets = []

    def log_preds(self, y_pred_prob, y):
        self.pred_probs.append(y_pred_prob)
        self.targets.append(y)

    def on_validation_epoch_end(self):
        targets = torch.cat(self.targets, dim=0).cpu().view(-1)
        pred_prob = torch.cat(self.pred_probs, dim=0).cpu()
        preds = torch.argmax(pred_prob, dim=1)

        self._reset_aggregates()

        # cross_entropy_loss = skm.log_loss(targets, pred_prob)

        log = {
            'val_acc_epoch': skm.accuracy_score(targets, preds),
            'epoch': self.current_epoch,
            # 'val_xent': cross_entropy_loss
        }
        print(log)

        if self.first_epoch:
            log['true_distr'] = wandb.Histogram(targets)
            self.first_epoch = False

        self.logger.experiment.log(log)

class AnswerPredTask(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()  # save config
        self.config = config  # config is a dotmap

        self.first_epoch = True
        self._reset_aggregates()

    def _reset_aggregates(self):
        self.pred_probs = []
        self.targets = []

    def log_preds(self, y_pred_prob, y):
        self.pred_probs.append(y_pred_prob)
        self.targets.append(y)

    def on_validation_epoch_end(self):
        targets = torch.cat(self.targets, dim=0).cpu().view(-1)
        pred_prob = torch.cat(self.pred_probs, dim=0).cpu()
        preds = torch.argmax(pred_prob, dim=1)

        # make sure our masking is successful
        for i in range(targets.shape[0]):
            assert targets[i] == 0 or targets[i] == 1, f"target is {targets[i]}"
        # we don't predict -1, and don't train on them

        self._reset_aggregates()

        # we can predict ROC?
        # Target scores, can either be probability estimates of the positive
        #  class
        roc_pred_prob = pred_prob[:, :2]  # label index 3 is -1 (no prediction)
        # normalize
        roc_pred_prob = roc_pred_prob / torch.sum(roc_pred_prob, dim=1, keepdim=True)
        roc_pred_prob = roc_pred_prob[:, 1]

        log = {
            # 'val_correctness_acc': skm.accuracy_score(filtered_targets, filtered_preds),
            'val_acc_epoch': skm.accuracy_score(targets, preds),
            'val_roc_epoch': skm.roc_auc_score(targets, roc_pred_prob),
            'epoch': self.current_epoch,
        }
        print(log)

        if self.first_epoch:
            log['true_distr'] = wandb.Histogram(targets)
            self.first_epoch = False

        self.logger.experiment.log(log)

class ScorePredTask(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()  # save config
        self.config = config  # config is a dotmap

        self.first_epoch = True
        self._reset_aggregates()

    def _reset_aggregates(self):
        self.pred_probs = []
        self.targets = []

    def log_preds(self, y_pred_prob, y):
        self.pred_probs.append(y_pred_prob)
        self.targets.append(y)

    def on_validation_epoch_end(self):
        # (total_size, 8)
        targets = torch.cat(self.targets, dim=0).cpu().view(-1)
        preds = torch.cat(self.pred_probs, dim=0).cpu().view(-1)

        self._reset_aggregates()

        # we predict mean absolute error

        log = {
            # 'val_correctness_acc': skm.accuracy_score(filtered_targets, filtered_preds),
            'val_mae_epoch': skm.mean_absolute_error(targets, preds),
            'epoch': self.current_epoch,
        }
        print(log)

        if self.first_epoch:
            log['true_distr'] = wandb.Histogram(targets)
            self.first_epoch = False

        self.logger.experiment.log(log)

class JointPredTask(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()  # save config
        self.config = config  # config is a dotmap

        # self.first_epoch = True
        self._reset_aggregates()

    def _reset_aggregates(self):
        self.state_pred_probs = []
        self.state_targets = []

        self.answer_pred_probs = []
        self.answer_targets = []

        self.score_pred_probs = []
        self.score_targets = []

    def log_preds(self, y_pred_prob, y, category):
        assert category in {'state', 'answer', 'score'}
        eval("self."+category+"_pred_probs").append(y_pred_prob)
        eval("self."+category+"_targets").append(y)
        # self.pred_probs.append(y_pred_prob)
        # self.targets.append(y)

    def on_validation_epoch_end(self):
        # start with state
        targets = torch.cat(self.state_targets, dim=0).cpu().view(-1)
        pred_prob = torch.cat(self.state_pred_probs, dim=0).cpu()
        preds = torch.argmax(pred_prob, dim=1)

        # cross_entropy_loss = skm.log_loss(targets, pred_prob)

        log = {
            'state_val_acc_epoch': skm.accuracy_score(targets, preds),
            'epoch': self.current_epoch,
        }

        # then answer

        targets = torch.cat(self.answer_targets, dim=0).cpu().view(-1)
        pred_prob = torch.cat(self.answer_pred_probs, dim=0).cpu()
        preds = torch.argmax(pred_prob, dim=1)

        # make sure our masking is successful
        for i in range(targets.shape[0]):
            assert targets[i] == 0 or targets[i] == 1, f"target is {targets[i]}"
        # we don't predict -1, and don't train on them

        roc_pred_prob = pred_prob[:, :2]  # label index 3 is -1 (no prediction)
        # normalize
        roc_pred_prob = roc_pred_prob / torch.sum(roc_pred_prob, dim=1, keepdim=True)
        roc_pred_prob = roc_pred_prob[:, 1]  # the probability of "correct"

        log['answer_val_acc_epoch'] = skm.accuracy_score(targets, preds)
        log['answer_val_roc_epoch'] = skm.roc_auc_score(targets, roc_pred_prob)
        log['answer_val_rmse_epoch'] = np.sqrt(skm.mean_squared_error(targets.float(), roc_pred_prob))

        # then scores
        targets = torch.cat(self.score_targets, dim=0).cpu().view(-1)
        preds = torch.cat(self.score_pred_probs, dim=0).cpu().view(-1)

        # we predict mean absolute error
        log['score_val_mae_epoch'] = skm.mean_absolute_error(targets, preds)

        print(log)
        self._reset_aggregates()

        self.logger.experiment.log(log)