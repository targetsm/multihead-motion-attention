"""
Neural networks for motion prediction.

Copyright ETH Zurich, Manuel Kaufmann
"""
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from data import AMASSBatch
from losses import mse
from model import GCN, util


def create_model(config):
    # This is a helper function that can be useful if you have several model definitions that you want to
    # choose from via the command line. For now, we just return the Dummy model.
    return MultiHeadAttModel(config)


class BaseModel(nn.Module):
    """A base class for neural networks that defines an interface and implements a few common functions."""

    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.pose_size = config.pose_size
        self.create_model()

    # noinspection PyAttributeOutsideInit
    def create_model(self):
        """Create the model, called automatically by the initializer."""
        raise NotImplementedError("Must be implemented by subclass.")

    def forward(self, batch: AMASSBatch):
        """The forward pass."""
        raise NotImplementedError("Must be implemented by subclass.")

    def backward(self, batch: AMASSBatch, model_out):
        """The backward pass."""
        raise NotImplementedError("Must be implemented by subclass.")

    def model_name(self):
        """A summary string of this model. Override this if desired."""
        return '{}-lr{}'.format(self.__class__.__name__, self.config.lr)


class DummyModel(BaseModel):
    """
    This is a dummy model. It provides basic implementations to demonstrate how more advanced models can be built.
    """

    def __init__(self, config):
        self.n_history = 10
        super(DummyModel, self).__init__(config)

    # noinspection PyAttributeOutsideInit
    def create_model(self):
        # In this model we simply feed the last time steps of the seed to a dense layer and
        # predict the targets directly.
        self.dense = nn.Linear(in_features=self.n_history * self.pose_size,
                               out_features=self.config.target_seq_len * self.pose_size)

    def forward(self, batch: AMASSBatch):
        """
        The forward pass.
        :param batch: Current batch of data.
        :return: Each forward pass must return a dictionary with keys {'seed', 'predictions'}.
        """
        model_out = {'seed': batch.poses[:, :self.config.seed_seq_len],
                     'predictions': None}
        batch_size = batch.batch_size
        model_in = batch.poses[:, self.config.seed_seq_len-self.n_history:self.config.seed_seq_len]
        pred = self.dense(model_in.reshape(batch_size, -1))
        model_out['predictions'] = pred.reshape(batch_size, self.config.target_seq_len, -1)
        return model_out

    def backward(self, batch: AMASSBatch, model_out):
        """
        The backward pass.
        :param batch: The same batch of data that was passed into the forward pass.
        :param model_out: Whatever the forward pass returned.
        :return: The loss values for book-keeping, as well as the targets for convenience.
        """
        predictions = model_out['predictions']
        targets = batch.poses[:, self.config.seed_seq_len:]

        total_loss = mse(predictions, targets)

        # If you have more than just one loss, just add them to this dict and they will automatically be logged.
        loss_vals = {'total_loss': total_loss.cpu().item()}

        if self.training:
            # We only want to do backpropagation in training mode, as this function might also be called when evaluating
            # the model on the validation set.
            total_loss.backward()

        return loss_vals, targets

class MultiHeadStackedModel(BaseModel):

    def __init__(self, config):
        super(MultiHeadStackedModel, self).__init__(config)

        print(config)
        in_features = 135
        kernel_size = 10  # M
        num_stage = 4
        dct_n = 24
        d_model = 128

        self.kernel_size = kernel_size
        self.d_model = d_model
        self.dct_n = dct_n

        self.convQ = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        self.convK = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        self.gcn = GCN.GCN(input_feature=(dct_n) * 2, hidden_feature=512, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features)

        self.multihead_attn = nn.MultiheadAttention(self.d_model, 16).to('cuda:0')

        self.ff_nn = nn.Sequential(
            nn.Linear(in_features=(in_features * dct_n), out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=d_model),
            nn.ReLU(),
        )

        self.rev_ff_nn = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=(in_features * dct_n)),
            nn.ReLU(),
        )

        print(in_features * kernel_size)


    def create_model(self):
        pass

    def forward(self, batch: AMASSBatch):
        """
        The forward pass.
        :param batch: Current batch of data.
        :return: Each forward pass must return a dictionary with keys {'seed', 'predictions'}.
        """

        model_out = {'seed': batch.poses[:, :self.config.seed_seq_len],
                     'predictions': None}
        batch_size = batch.batch_size
        src = batch.poses
        output_n = 24  # number of output frames T
        input_n = 120  # number of input frames N

        dct_n = self.dct_n
        src = src[:, :input_n]  # [bs,in_n,dim]
        src_tmp = src.clone()
        bs = src.shape[0]
        src_key_tmp = src_tmp.transpose(1, 2)[:, :, :(input_n - output_n)].clone()
        src_query_tmp = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()

        dct_m, idct_m = util.get_dct_matrix(self.kernel_size + output_n)
        dct_m = torch.from_numpy(dct_m).float().cuda()
        idct_m = torch.from_numpy(idct_m).float().cuda()

        vn = input_n - self.kernel_size - output_n + 1
        vl = self.kernel_size + output_n
        idx = np.expand_dims(np.arange(vl), axis=0) + \
              np.expand_dims(np.arange(vn), axis=1)
        src_value_tmp = src_tmp[:, idx].clone().reshape(
            [bs * vn, vl, -1])
        src_value_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_value_tmp).reshape(
            [bs, vn, dct_n, -1]).transpose(2, 3).reshape(
            [bs, vn, -1])  # 

        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n
        outputs = []

        key_tmp = self.convK(src_key_tmp / 1000.0).transpose(0,1).transpose(0,2)

        query_tmp = self.convQ(src_query_tmp / 1000.0).transpose(0,1).transpose(0,2)
        value_tmp = self.ff_nn(src_value_tmp).transpose(1,2).transpose(0,1).transpose(0,2)

        mh_attn_out, weights = self.multihead_attn(query_tmp, key_tmp, value_tmp)

        af_ff_out = self.rev_ff_nn(mh_attn_out).transpose(0,1).reshape(batch_size, -1, dct_n)

        input_gcn = src_tmp[:, idx]  # shape:[16, 34, 135]

        dct_in_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)

        dct_in_tmp = torch.cat([dct_in_tmp, af_ff_out], dim=-1)

        dct_out_tmp = self.gcn(dct_in_tmp)
        out_gcn = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),
                                dct_out_tmp[:, :, :dct_n].transpose(1, 2))
        outputs.append(out_gcn.unsqueeze(2))

        outputs = torch.cat(outputs, dim=2)

        out_sq = outputs.squeeze()
        out_sq = out_sq[:, -24:, :]
        model_out['predictions'] = out_sq.reshape(batch_size, self.config.target_seq_len, -1)
        return model_out



    def backward(self, batch: AMASSBatch, model_out):
        """
        The backward pass.
        :param batch: The same batch of data that was passed into the forward pass.
        :param model_out: Whatever the forward pass returned.
        :return: The loss values for book-keeping, as well as the targets for convenience.
        """
        predictions = model_out['predictions']
        targets = batch.poses[:, -24:]

        total_loss = mse(predictions, targets)

        # If you have more than just one loss, just add them to this dict and they will automatically be logged.
        loss_vals = {'total_loss': total_loss.cpu().item()}

        if self.training:
            # We only want to do backpropagation in training mode, as this function might also be called when evaluating
            # the model on the validation set.
            total_loss.backward()

        return loss_vals, targets



class MultiHeadAttModel(BaseModel):
    """
    Code based on https://github.com/wei-mao-2019/HisRepItself
    """

    def __init__(self, config):
        super(MultiHeadAttModel, self).__init__(config)

        # __init__ arguments in the original attention model
        in_features = 135  # 48
        kernel_size = 10
        d_model = 512
        num_stage = 4
        dct_n = 34

        self.kernel_size = kernel_size
        self.d_model = d_model
        # self.seq_in = seq_in
        self.dct_n = dct_n


        self.convQ = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        self.convK = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        self.convQ1 = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        self.convK1 = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        self.convQ2 = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        self.convK2 = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        self.convQ3 = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        self.convK3 = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        self.gcn = GCN.GCN(input_feature=(dct_n) * 5, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features)


    def create_model(self):
        pass

    def forward(self, batch: AMASSBatch):
        """
        The forward pass.
        :param batch: Current batch of data.
        :return: Each forward pass must return a dictionary with keys {'seed', 'predictions'}.
        """

        model_out = {'seed': batch.poses[:, :self.config.seed_seq_len],
                     'predictions': None}
        batch_size = batch.batch_size
        #model_in = batch.poses[:, self.config.seed_seq_len - self.n_history:self.config.seed_seq_len]
        model_in = batch.poses
        # old forward docstring and variables
        """
        :param src: [batch_size,seq_len,feat_dim]
        :param output_n:
        :param input_n:
        :param frame_n:
        :param dct_n:
        :param itera:
        :return:
        """
        src = model_in
        output_n = 24
        input_n = 120
        itera = 1

        dct_n = self.dct_n
        src = src[:, :input_n]  # [bs,in_n,dim]
        src_tmp = src.clone()
        bs = src.shape[0]
        src_key_tmp = src_tmp.transpose(1, 2)[:, :, :(input_n - output_n)].clone()
        src_query_tmp = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()

        src_key_tmp1 = src_tmp.transpose(1, 2)[:, :, :(input_n - output_n)].clone()
        src_query_tmp1 = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()

        src_key_tmp2 = src_tmp.transpose(1, 2)[:, :, :(input_n - output_n)].clone()
        src_query_tmp2 = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()

        src_key_tmp3 = src_tmp.transpose(1, 2)[:, :, :(input_n - output_n)].clone()
        src_query_tmp3 = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()

        dct_m, idct_m = util.get_dct_matrix(self.kernel_size + output_n)
        dct_m = torch.from_numpy(dct_m).float().cuda()
        idct_m = torch.from_numpy(idct_m).float().cuda()

        vn = input_n - self.kernel_size - output_n + 1
        vl = self.kernel_size + output_n
        idx = np.expand_dims(np.arange(vl), axis=0) + \
              np.expand_dims(np.arange(vn), axis=1)
        src_value_tmp = src_tmp[:, idx].clone().reshape(
            [bs * vn, vl, -1])
        src_value_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_value_tmp).reshape(
            [bs, vn, dct_n, -1]).transpose(2, 3).reshape(
            [bs, vn, -1])  # [32,40,66*11]

        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n
        outputs = []

        key_tmp = self.convK(src_key_tmp / 1000.0)
        key_tmp1 = self.convK1(src_key_tmp1 / 1000.0)
        key_tmp2 = self.convK2(src_key_tmp2 / 1000.0)
        key_tmp3 = self.convK3(src_key_tmp3 / 1000.0)

        query_tmp = self.convQ(src_query_tmp / 1000.0)
        query_tmp1 = self.convQ1(src_query_tmp1 / 1000.0)
        query_tmp2 = self.convQ2(src_query_tmp2 / 1000.0)
        query_tmp3 = self.convQ3(src_query_tmp3 / 1000.0)

        score_tmp = torch.matmul(query_tmp.transpose(1, 2), key_tmp) + 1e-15
        score_tmp1 = torch.matmul(query_tmp1.transpose(1, 2), key_tmp1) + 1e-15
        score_tmp2 = torch.matmul(query_tmp2.transpose(1, 2), key_tmp2) + 1e-15
        score_tmp3 = torch.matmul(query_tmp3.transpose(1, 2), key_tmp3) + 1e-15

        att_tmp = score_tmp / (torch.sum(score_tmp, dim=2, keepdim=True))
        att_tmp1 = score_tmp1 / (torch.sum(score_tmp, dim=2, keepdim=True))
        att_tmp2 = score_tmp2 / (torch.sum(score_tmp, dim=2, keepdim=True))
        att_tmp3 = score_tmp3 / (torch.sum(score_tmp, dim=2, keepdim=True))

        dct_att_tmp = torch.matmul(att_tmp, src_value_tmp)[:, 0].reshape(
            [bs, -1, dct_n])

        dct_att_tmp1 = torch.matmul(att_tmp1, src_value_tmp)[:, 0].reshape(
            [bs, -1, dct_n])

        dct_att_tmp2 = torch.matmul(att_tmp2, src_value_tmp)[:, 0].reshape(
            [bs, -1, dct_n])

        dct_att_tmp3 = torch.matmul(att_tmp3, src_value_tmp)[:, 0].reshape(
            [bs, -1, dct_n])

        input_gcn = src_tmp[:, idx]
        dct_in_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)
        dct_in_tmp = torch.cat([dct_in_tmp, dct_att_tmp, dct_att_tmp1, dct_att_tmp2, dct_att_tmp3], dim=-1)
        dct_out_tmp = self.gcn(dct_in_tmp)
        out_gcn = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),
                                dct_out_tmp[:, :, :dct_n].transpose(1, 2))
        outputs.append(out_gcn.unsqueeze(2))


        outputs = torch.cat(outputs, dim=2)
        #print(outputs.shape)
        #return outputs

        #pred = self.dense(model_in.reshape(batch_size, -1))
        out_sq = outputs.squeeze()
        out_sq = out_sq[:, -24:, :]
        model_out['predictions'] = out_sq.reshape(batch_size, self.config.target_seq_len, -1)
        #print(model_out['predictions'].shape)
        return model_out

    def backward(self, batch: AMASSBatch, model_out):
        """
        The backward pass.
        :param batch: The same batch of data that was passed into the forward pass.
        :param model_out: Whatever the forward pass returned.
        :return: The loss values for book-keeping, as well as the targets for convenience.
        """
        predictions = model_out['predictions']
        targets = batch.poses[:, -24:]
        #print(targets.shape)
        total_loss = mse(predictions, targets)

        # If you have more than just one loss, just add them to this dict and they will automatically be logged.
        loss_vals = {'total_loss': total_loss.cpu().item()}

        if self.training:
            # We only want to do backpropagation in training mode, as this function might also be called when evaluating
            # the model on the validation set.
            total_loss.backward()

        return loss_vals, targets
