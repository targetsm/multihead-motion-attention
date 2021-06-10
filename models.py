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
    return CustomTransformer(config)


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


class MultiHeadModel(BaseModel):

    def __init__(self, config):
        super(MultiHeadModel, self).__init__(config)

        print(config)
        in_features = 135
        kernel_size = 10  # M
        num_stage = 2
        dct_n = 12
        d_model = dct_n * in_features

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

        self.gcn = GCN.GCN(input_feature=(dct_n) * 2, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features)

        self.multihead_attn = nn.MultiheadAttention(self.d_model, 10).to('cuda:0')

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
        itera = 1

        # construct keys and queries
        dct_n = self.dct_n
        src = src[:, :input_n]  # remove given output frames
        src_tmp = src.clone()  # shape:[16, 120, 135]=[batch_size, input_n, in_features]
        bs = src.shape[0]  # batch_size
        src_key_tmp = src_tmp.transpose(1, 2)[:, :, :(input_n - output_n)].clone()  # construct keys, shape:[16, 135, 96]
        src_query_tmp = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()  # construct queries, shape:[16, 135, 10]

        # get dct matrices
        dct_m, idct_m = util.get_dct_matrix(self.kernel_size + output_n)
        dct_m = torch.from_numpy(dct_m).float().cuda()
        idct_m = torch.from_numpy(idct_m).float().cuda()

        # construct values and apply cosine transform
        vn = input_n - self.kernel_size - output_n + 1  # N - M - T + 1 = 87
        vl = self.kernel_size + output_n  # M + T = 34
        idx = np.expand_dims(np.arange(vl), axis=0) + \
              np.expand_dims(np.arange(vn), axis=1)
        src_value_tmp = src_tmp[:, idx].clone().reshape(
            [bs * vn, vl, -1])  # shape:[1392, 34, 135]
        src_value_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_value_tmp).reshape(
            [bs, vn, dct_n, -1]).transpose(2, 3).reshape(
            [bs, vn, -1])  # shape:[16, 87, 34*135]

        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n
        outputs = []

        key_tmp = self.convK(src_key_tmp / 1000.0)  # shape:[16, 512, 87] input embedding
        for i in range(itera):
            # Motion Attention
            query_tmp = self.convQ(src_query_tmp / 1000.0)  # shape:[16, 512, 1] input embedding
            query_tmp = query_tmp.reshape([1, bs, self.d_model])
            key_tmp = key_tmp.reshape([87, bs, self.d_model])
            src_value_tmp = src_value_tmp.reshape([87, bs, self.d_model])
            dct_att_tmp, attn_output_weights = self.multihead_attn(query_tmp, key_tmp, src_value_tmp)
            dct_att_tmp = (dct_att_tmp.transpose(0,1))[:, 0].reshape(
                [bs, -1, dct_n])
            input_gcn = src_tmp[:, idx]  # shape:[16, 34, 135]
            dct_in_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)
            dct_in_tmp = torch.cat([dct_in_tmp, dct_att_tmp], dim=-1)
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


class TransformerModel(BaseModel):
    def __init__(self, config):
        super(TransformerModel, self).__init__(config)

        self.transformer = torch.nn.Transformer(d_model=135,
                                                nhead=15,
                                                num_encoder_layers=6,
                                                num_decoder_layers=6,
                                                dim_feedforward=512,
                                                dropout=0.1,
                                                activation='relu',
                                                custom_encoder=None,
                                                custom_decoder=None)

    def forward(self, batch: AMASSBatch):
        model_out = {'seed': batch.poses[:, :self.config.seed_seq_len], 'predictions': None}
        # transpose the Batch such that the batch size in the middle
        src = batch.poses
        src = src[:, :120]

        input_n = 120
        output_n = 24
        self.kernel_size = 10
        vn = input_n - self.kernel_size - output_n + 1  # N - M - T + 1 = 87
        vl = self.kernel_size + output_n  # M + T = 34
        idx = np.expand_dims(np.arange(vl), axis=0) + \
              np.expand_dims(np.arange(vn), axis=1)

        src = src[:, idx].reshape(
            [16 * vn, vl, -1])

        swapped = src.transpose(0,1)

        out = self.transformer(swapped[:,:,:], swapped[-24:,:,:])
        out = out.transpose(0,1)

        model_out['predictions'] = out
        return model_out

    def create_model(self):
        pass

    def backward(self, batch: AMASSBatch, model_out):
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


class CustomTransformer(BaseModel):

    def __init__(self, config):
        super(CustomTransformer, self).__init__(config)

        print(config)
        in_features = 135
        kernel_size = 10  # M
        dct_n = 144
        d_model = dct_n * in_features
        self.output_n = 24

        self.kernel_size = kernel_size
        self.d_model = d_model
        self.dct_n = dct_n

        self.transformer = torch.nn.Transformer(d_model=in_features,
                                                nhead=15,
                                                num_encoder_layers=6,
                                                num_decoder_layers=6,
                                                dim_feedforward=512,
                                                dropout=0.1,
                                                activation='relu',
                                                custom_encoder=None,
                                                custom_decoder=None)

        self.tgt_mask = self.transformer.generate_square_subsequent_mask(self.output_n).cuda()

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
        bs = batch.batch_size
        src = batch.poses
        output_n = 24  # number of output frames T
        input_n = 120  # number of input frames N

        if src.shape[1] == 120 or not self.training:
            # initialized the input of the decoder with sos_idx (start of sentence token idx)
            src_encoder = src[:, :120].transpose(0,1)
            encoder_output = self.transformer.encoder(src_encoder)
            output = src[:, -1:, :].clone().cuda().transpose(0,1)
            for t in range(1, output_n):
                output = output[:t]
                tgt_mask_t = self.tgt_mask[:t,:t]
                decoder_output = self.transformer.decoder(tgt=output,
                                         memory=encoder_output,
                                         tgt_mask=tgt_mask_t).cuda()
                output = torch.cat([output, decoder_output], dim=0)
            model_out['predictions'] = output.transpose(0, 1)[:, -24:]
            return model_out

        src_tmp = src.clone()  # torch.Size([16, 144, 135])
        src_transformer = src_tmp[:, :input_n, :].clone().transpose(0,1)  # torch.Size([16, 120, 135])
        tgt_transformer = src_tmp[:, -output_n:, :].clone().transpose(0,1) # torch.Size([16, 34, 135])

        output = self.transformer(src_transformer, tgt_transformer, tgt_mask=self.tgt_mask).cuda()

        model_out['predictions'] = output.transpose(0,1)
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

class CustomTransformer2(BaseModel):

    def __init__(self, config):
        super(CustomTransformer2, self).__init__(config)

        print(config)
        in_features = 135
        kernel_size = 10  # M
        dct_n = 144
        d_model = dct_n * in_features

        self.kernel_size = kernel_size
        self.d_model = d_model
        self.dct_n = dct_n

        self.transformer = torch.nn.Transformer(d_model=in_features,
                                                nhead=15,
                                                num_encoder_layers=6,
                                                num_decoder_layers=6,
                                                dim_feedforward=512,
                                                dropout=0.1,
                                                activation='relu',
                                                custom_encoder=None,
                                                custom_decoder=None)

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
        bs = batch.batch_size
        src = batch.poses
        output_n = 24  # number of output frames T
        input_n = 120  # number of input frames N
        itera = 1

        src_tmp = src.clone()  # torch.Size([16, 144, 135])
        src_transformer = src_tmp[:, :input_n, :].clone()  # torch.Size([16, 120, 135])
        tgt_transformer = src_tmp[:, :, :].clone()  # torch.Size([16, 34, 135])

        src_transformer = src_transformer[:, list(range(0,120)) + [-1]*output_n]
        if tgt_transformer.shape[1] == 120:
            tgt_transformer = tgt_transformer[:, list(range(0, 120)) + [-1] * output_n]

        # get dct matrices
        dct_m, idct_m = util.get_dct_matrix(input_n + output_n)
        dct_m = torch.from_numpy(dct_m).float().cuda()
        idct_m = torch.from_numpy(idct_m).float().cuda()
        src_transformer = torch.matmul(dct_m[:self.dct_n].unsqueeze(dim=0), src_transformer).transpose(0,1)

        tgt_transformer = torch.matmul(dct_m[:self.dct_n].unsqueeze(dim=0), tgt_transformer).transpose(0,1)

        output = self.transformer(src_transformer, tgt_transformer).cuda()

        output = torch.matmul(idct_m[:, :self.dct_n].unsqueeze(dim=0), output.transpose(0,1))
        model_out['predictions'] = output[:, -24:, :]
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

class CustomTransformer3(BaseModel):

    def __init__(self, config):
        super(CustomTransformer3, self).__init__(config)

        print(config)
        in_features = 135
        kernel_size = 10  # M
        dct_n = 12
        d_model = dct_n * in_features

        self.kernel_size = kernel_size
        self.d_model = d_model
        self.dct_n = dct_n

        self.transformer = torch.nn.Transformer(d_model=in_features,
                                                nhead=15,
                                                num_encoder_layers=6,
                                                num_decoder_layers=6,
                                                dim_feedforward=512,
                                                dropout=0.1,
                                                activation='relu',
                                                custom_encoder=None,
                                                custom_decoder=None)

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
        bs = batch.batch_size
        src = batch.poses
        output_n = 24  # number of output frames T
        input_n = src.shape[1] - output_n  # number of input frames N
        itera = 1

        src_tmp = src.clone()  # torch.Size([16, 144, 135])
        src_transformer = src_tmp[:, :, :].clone()  # torch.Size([16, 120, 135])
        tgt_transformer = src_tmp[:, :, :].clone()  # torch.Size([16, 34, 135])

        vn = input_n - self.kernel_size + 1  # N - M - T + 1 = 87
        vl_src = self.kernel_size  # M + T = 34
        vl_tgt = self.kernel_size + output_n  # M + T = 34
        idx_src = np.expand_dims(np.arange(vl_src), axis=0) + \
                  np.expand_dims(np.arange(vn), axis=1)
        idx_tgt = np.expand_dims(np.arange(vl_tgt), axis=0) + \
                  np.expand_dims(np.arange(vn), axis=1)

        src_transformer = src_transformer[:, idx_src].reshape(
            [bs * vn, vl_src, -1])
        src_transformer = src_transformer[:, list(range(0, vl_src)) + [-1] * output_n]
        tgt_transformer = tgt_transformer[:, idx_tgt].reshape(
            [bs * vn, vl_tgt, -1])

        # get dct matrices
        dct_m, idct_m = util.get_dct_matrix(self.kernel_size + output_n)
        dct_m = torch.from_numpy(dct_m).float().cuda()
        idct_m = torch.from_numpy(idct_m).float().cuda()
        src_transformer = torch.matmul(dct_m[:self.dct_n].unsqueeze(dim=0), src_transformer)
        tgt_transformer = torch.matmul(dct_m[:self.dct_n].unsqueeze(dim=0), tgt_transformer)

        output = self.transformer(src_transformer, tgt_transformer).cuda()
        output = torch.matmul(idct_m[:, :self.dct_n].unsqueeze(dim=0), output)
        idx = [x * vn for x in range(0, bs)]
        model_out['predictions'] = output[idx, -24:, :]
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


class CustomTransformer4(BaseModel):

    def __init__(self, config):
        super(CustomTransformer4, self).__init__(config)

        print(config)
        in_features = 135
        kernel_size = 10  # M
        dct_n = 12
        d_model = dct_n * in_features

        self.kernel_size = kernel_size
        self.d_model = d_model
        self.dct_n = dct_n

        self.transformer = torch.nn.Transformer(d_model=d_model,
                                                nhead=15,
                                                num_encoder_layers=6,
                                                num_decoder_layers=6,
                                                dim_feedforward=512,
                                                dropout=0.1,
                                                activation='relu',
                                                custom_encoder=None,
                                                custom_decoder=None)

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
        bs = batch.batch_size
        src = batch.poses
        output_n = 24  # number of output frames T
        input_n = 120  # number of input frames N
        itera = 1

        src_tmp = src.clone()  # torch.Size([16, 144, 135])

        src_transformer = src_tmp[:, list(range(0, input_n)) + [-1] * output_n].clone()  # torch.Size([16, 120, 135])
        tgt_transformer = src_tmp[:, :, :].clone()  # torch.Size([16, 34, 135])
        if src.shape[1] == 120:
            tgt_transformer = src_tmp[:, list(range(0, input_n)) + [-1] * output_n].clone()

        vn = input_n - self.kernel_size + 1  # N - M - T + 1 = 87
        vl = self.kernel_size + output_n  # M + T = 34
        idx_src = np.expand_dims(np.arange(vl), axis=0) + \
                  np.expand_dims(np.arange(vn), axis=1)
        idx_tgt = np.expand_dims(np.arange(vl), axis=0) + \
                  np.expand_dims(np.arange(vn), axis=1)
        src_transformer = src_transformer[:, idx_src].reshape(
            [bs * vn, vl, -1])
        tgt_transformer = tgt_transformer[:, idx_tgt].reshape(
            [bs * vn, vl, -1])

        # get dct matrices
        dct_m, idct_m = util.get_dct_matrix(self.kernel_size + output_n)
        dct_m = torch.from_numpy(dct_m).float().cuda()
        idct_m = torch.from_numpy(idct_m).float().cuda()

        src_transformer = torch.matmul(dct_m[:self.dct_n].unsqueeze(dim=0), src_transformer)
        src_transformer = src_transformer.reshape(
            [bs, vn, self.dct_n, -1]).transpose(2, 3).reshape(
            [bs, vn, -1])  # shape:[16, 87, 34*135]
        tgt_transformer = torch.matmul(dct_m[:self.dct_n].unsqueeze(dim=0), tgt_transformer)
        tgt_transformer = tgt_transformer.reshape(
            [bs, vn, self.dct_n, -1]).transpose(2, 3).reshape(
            [bs, vn, -1])  # shape:[16, 87, 34*135]

        src_transformer = src_transformer.transpose(0,1)
        tgt_transformer = tgt_transformer.transpose(0,1)
        output = self.transformer(src_transformer, tgt_transformer).cuda()
        output = output.transpose(0,1)

        output = output.reshape([bs*vn, self.dct_n, 135])
        output = torch.matmul(idct_m[:, :self.dct_n].unsqueeze(dim=0), output)
        idx = [x * vn for x in range(0, bs)]
        model_out['predictions'] = output[idx, -24:, :]
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




class MultiHeadStackedModel(BaseModel):

    def __init__(self, config):
        super(MultiHeadStackedModel, self).__init__(config)

        print(config)
        in_features = 135
        kernel_size = 10  # M
        num_stage = 2
        dct_n = 34
        d_model = 512

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

        self.gcn = GCN.GCN(input_feature=(dct_n) * 2, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features)

        self.multihead_attn = nn.MultiheadAttention(self.d_model, 16).to('cuda:0')
        # self.multihead_attn_layer2 = nn.MultiheadAttention(self.d_model, 16).to('cuda:0')
        #
        # self.neural_net = nn.Sequential(
        #     nn.Linear(in_features=d_model, out_features=d_model),
        #     nn.ReLU(),
        #     nn.Linear(in_features=d_model, out_features=d_model),
        #     nn.ReLU(),
        # )

        self.ff_nn = nn.Sequential(
            nn.Linear(in_features=(in_features * dct_n), out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=d_model),
            nn.ReLU(),
        )

        self.rev_ff_nn = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=(in_features * dct_n)),
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

        key_tmp = self.convK(src_key_tmp / 1000.0).swapaxes(0,1).swapaxes(0,2)

        query_tmp = self.convQ(src_query_tmp / 1000.0).swapaxes(0,1).swapaxes(0,2)
        value_tmp = self.ff_nn(src_value_tmp).swapaxes(1,2).swapaxes(0,1).swapaxes(0,2)

        mh_attn_out, weights = self.multihead_attn(query_tmp, key_tmp, value_tmp)

        # mh2_query_tmp = self.neural_net(mh_attn_out)
        # mh2_attn_out, weights = self.multihead_attn_layer2(mh2_query_tmp, key_tmp, value_tmp)

        af_ff_out = self.rev_ff_nn(mh_attn_out).swapaxes(0,1).reshape(batch_size, -1, dct_n)

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