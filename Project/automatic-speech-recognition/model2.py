# based on DeepSpeech

import torch
import MCVmetrics
import MCVencoding

class SequenceWise(torch.nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class MaskConv(torch.nn.Module):
    def __init__(self, seq_module):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that the
        results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (BxCxDxT)
        :param seq_module: The sequential module containing the conv stack.
        """
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:
            x = module(x)
            mask = torch.BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths


class BatchRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=torch.nn.LSTM, blank_label=27, bidirectional=True, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.blank_label = blank_label
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(torch.nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, output_lengths)
        x, h = self.rnn(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, padding_value=self.blank_label)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        return x


class InferenceSoftmax(torch.nn.Module):
    def forward(self, input_):
        return torch.nn.functional.softmax(input_, dim=-1)


class TrainingSoftmax(torch.nn.Module):
    def forward(self, input_):
        return input_.transpose(0,1).log_softmax(-1)


class DeepSpeech(torch.nn.Module):

    lstm_dropout = 0.0
    lstm_hidden_size = 1024
    lstm_layers = 3
    lstm_bidirectional = True

    adam_learning_rate = 2.5e-5
    adam_betas = (0.9, 0.999)
    adam_eps = 1e-8
    adam_weight_decay = 1e-5
    adam_learning_anneal = 0.99
    abs_path = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Project/automatic-speech-recognition/model-params/'

    def __init__(self, num_features=64, num_classes=28, blank_label=27, load_path_ext='asr-DS.pth',save_path_ext='asr-DS.pth'):
        self.num_classes = num_classes
        self.num_features = num_features
        self.blank_label = blank_label
        self.model_load_path = self.abs_path+load_path_ext
        self.model_save_path = self.abs_path+save_path_ext
        super().__init__()
        self.num_conv_features = 32*16
        self.rnns = torch.nn.Sequential(
            BatchRNN(
                input_size=self.num_conv_features,
                hidden_size=self.lstm_hidden_size,
                rnn_type=torch.nn.LSTM,
                bidirectional=self.lstm_bidirectional,
                blank_label=self.blank_label,
                batch_norm=False
            ),
            *(
                BatchRNN(
                    input_size=self.lstm_hidden_size,
                    hidden_size=self.lstm_hidden_size,
                    rnn_type=torch.nn.LSTM,
                    bidirectional=self.lstm_bidirectional,
                    blank_label=self.blank_label,
                ) for x in range(self.lstm_layers - 1)
            )
        )
        self.cnn = MaskConv(torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            torch.nn.BatchNorm2d(32),
            torch.nn.Hardtanh(0, 20, inplace=True),
            torch.nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            torch.nn.BatchNorm2d(32),
            torch.nn.Hardtanh(0, 20, inplace=True)
        ))
        self.training_softmax = TrainingSoftmax()
        self.inference_softmax = InferenceSoftmax()
        self.validation_decoder = MCVencoding.EncoderDecoder()
        self.clf = torch.nn.Sequential(
            # torch.nn.LayerNorm(normalized_shape=self.lstm_hidden_size),
            # torch.nn.BatchNorm1d(self.lstm_hidden_size),
            torch.nn.Linear(in_features=self.lstm_hidden_size, out_features=self.num_classes, bias=False)
        )
        self.classifier = SequenceWise(self.clf)
        self.criterion = torch.nn.CTCLoss(blank=self.blank_label, reduction='mean', zero_infinity=True)
        # self.optimizer = torch.optim.SGD(params=self.parameters(), lr=1e-3, momentum=0.9)
        self.optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.adam_learning_rate, betas=self.adam_betas, eps=self.adam_eps, weight_decay=self.adam_weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=self.adam_learning_anneal)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2, patience=5)

    def training_step(self, inputs, labels, input_lengths, label_lengths):
        self.optimizer.zero_grad()
        outputs,seq_lengths = self(inputs,input_lengths)
        outputs_dec = self.inference_softmax(outputs)
        outputs = self.training_softmax(outputs)
        loss = self.criterion(outputs, labels, seq_lengths, label_lengths)
        loss.backward()
        self.optimizer.step()
        return outputs_dec, loss

    def validation_step(self, inputs, labels, input_lengths, label_lengths):
        self.optimizer.zero_grad()
        outputs,seq_lengths = self(inputs,input_lengths)
        outputs_dec = self.inference_softmax(outputs)
        outputs = self.training_softmax(outputs)
        loss = self.criterion(outputs, labels, seq_lengths, label_lengths)
        return outputs_dec, loss

    def compute_evaluation_metrics(self, outputs, labels):
        out_pred = self.validation_decoder.greedy_output_decoding(outputs)
        out_true = self.validation_decoder.label_decoding(labels)
        batch_wer, wer_ref_lens = MCVmetrics.batch_wer(out_true, out_pred, ignore_case=True)
        batch_cer, cer_ref_lens = MCVmetrics.batch_cer(out_true, out_pred, ignore_case=True)
        batch_wavg_wer = MCVmetrics.weighted_avg_wer(batch_wer, wer_ref_lens)
        batch_wavg_cer = MCVmetrics.weighted_avg_cer(batch_cer, cer_ref_lens)
        return batch_wavg_wer, batch_wavg_cer

    def forward(self, x, lengths):
        output_lengths = self.get_seq_lens(lengths)
        x,_ = self.cnn(x,output_lengths)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1]*sizes[2], sizes[3])
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        for rnn in self.rnns:
            x = rnn(x, output_lengths)
        # x = x.transpose(0, 1)
        x = self.classifier(x)
        x = x.transpose(0,1)
        return x, output_lengths

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.cnn.modules():
            if type(m) == torch.nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) // m.stride[1] + 1)
        return seq_len.int()

    def load_saved_model(self):
        print(f'Loading model parameters from: {self.model_load_path}\n')
        return self.load_state_dict(state_dict=torch.load(self.model_load_path))

    def save_model(self):
        print(f'Saving model parameters to: {self.model_save_path}\n')
        torch.save(self.state_dict(),self.model_save_path)
