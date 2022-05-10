# select portions based on Deep Speech
# Deep Speech implementation available at https://github.com/SeanNaren/deepspeech.pytorch/tree/master/deepspeech_pytorch

import torch
import MCVmetrics
import MCVencoding

import kenlm
import ctcdecode

class RNN(torch.nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, dropout, blank_label=27, bidirectional=True):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.blank_label=blank_label
        self.rnn = torch.nn.LSTM(input_size=self.input_size, num_layers=self.num_layers, hidden_size=self.hidden_size, dropout=self.dropout, bidirectional=self.bidirectional, bias=True)

    def forward(self, x, output_lengths):
        x = torch.nn.utils.rnn.pack_padded_sequence(x, output_lengths, batch_first=False)
        x, _ = self.rnn(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, padding_value=self.blank_label, batch_first=False)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        return x

    def forward2(self, x):
        x, h = self.rnn(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        return x,h


class InferenceSoftmax(torch.nn.Module):
    def forward(self, input_):
        return torch.nn.functional.softmax(input_, dim=-1)


class TrainingSoftmax(torch.nn.Module):
    def forward(self, input_):
        return input_.transpose(0,1).log_softmax(-1)


class ASR(torch.nn.Module):

    kenlm_path = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Project/automatic-speech-recognition/kenlm/4gram_big.arpa'
    # kenlm_path = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Project/automatic-speech-recognition/kenlm/4gram_small.arpa'

    lstm_dropout = 0.2
    lstm_hidden_size = 1024
    lstm_layers = 3
    lstm_bidirectional = True

    adam_learning_rate = 2.5e-5
    adam_betas = (0.9, 0.999)
    adam_eps = 1e-8
    adam_weight_decay = 1e-5
    adam_learning_anneal = 0.99
    abs_path = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Project/automatic-speech-recognition/model-params/'

    def __init__(self, num_features=64, num_classes=28, blank_label=27, load_path_ext='asr-ASR.pth', save_path_ext='asr-ASR.pth'):
        self.num_classes = num_classes
        self.num_features = num_features
        self.blank_label = blank_label
        self.model_load_path = self.abs_path+load_path_ext
        self.model_save_path = self.abs_path+save_path_ext
        super().__init__()
        self.num_conv_features = 38*16
        self.cnn = torch.nn.Sequential(
            # shape = ( bs , 1 , 64 , length )
            torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1, padding=5//1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, stride=1, padding=5//1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=5//2),
            torch.nn.ReLU()
        )
        self.linear = torch.nn.Sequential(
            # assuming padding1=padding2=0, kernel1=kernel2=3, dilation1=dilation2=dilation3=1
            # in_features size = out_channels*(1 + (64-2-2)/stride3 + (2*padding3-kernel3)//stride3)
            # torch.nn.Linear(in_features=(16*58), out_features=256),
            torch.nn.Linear(in_features=(self.num_conv_features), out_features=self.num_conv_features),
            torch.nn.LayerNorm(normalized_shape=self.num_conv_features),
            torch.nn.ReLU(),
            # torch.nn.Dropout(p=self.p_dropout, inplace=False),
            torch.nn.Linear(in_features=self.num_conv_features, out_features=self.num_conv_features),
            torch.nn.LayerNorm(normalized_shape=self.num_conv_features),
            torch.nn.ReLU(),
            # torch.nn.Dropout(p=self.p_dropout, inplace=False),
            # torch.nn.Linear(in_features=256, out_features=256),
            # torch.nn.LayerNorm(normalized_shape=256),
            # torch.nn.ReLU()
        )
        self.lstm = RNN(input_size=self.num_conv_features,num_layers=self.lstm_layers,hidden_size=self.lstm_hidden_size,dropout=self.lstm_dropout,bidirectional=self.lstm_bidirectional)
        self.classifier = torch.nn.Sequential(
            # torch.nn.LayerNorm(normalized_shape=self.lstm_hidden_size),
            torch.nn.Linear(in_features=self.lstm_hidden_size, out_features=self.num_classes, bias=False),
        )
        self.training_softmax = TrainingSoftmax()
        self.inference_softmax = InferenceSoftmax()
        self.validation_decoder = MCVencoding.EncoderDecoder()
        # self.beam_decoder = ctcdecode.CTCBeamDecoder(labels=list("-abcdefghijklmnopqrstuvwxyz_"), blank_id=self.blank_label, log_probs_input=False, model_path=self.kenlm_path)
        self.beam_decoder = ctcdecode.CTCBeamDecoder(labels=list(" abcdefghijklmnopqrstuvwxyz_"), blank_id=self.blank_label, log_probs_input=False, model_path=self.kenlm_path)
        self.criterion = torch.nn.CTCLoss(blank=self.blank_label, reduction='mean', zero_infinity=True)
        self.optimizer = torch.optim.SGD(params=self.parameters(), lr=2.5e-4, momentum=0.9)
        # self.optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.adam_learning_rate, betas=self.adam_betas, eps=self.adam_eps, weight_decay=self.adam_weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2, patience=5)

    def forward(self, x, input_lengths):
        output_lengths = self.get_seq_lens(input_lengths)
        x = self.cnn(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1]*sizes[2], sizes[3])
        x = x.transpose(1,2).transpose(0,1).contiguous() 
        x = self.linear(x)
        x = self.lstm(x,output_lengths)
        # x = x.transpose(0, 1)
        x = self.classifier(x)
        x = x.transpose(0, 1)
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
        with torch.no_grad():
            outputs,seq_lengths = self(inputs,input_lengths)
            outputs_dec = self.inference_softmax(outputs)
            outputs = self.training_softmax(outputs)
            loss = self.criterion(outputs, labels, seq_lengths, label_lengths)
        return outputs_dec, loss

    def inference_step(self, inputs, input_lengths):
        with torch.no_grad():
            outputs, _ = self(inputs, input_lengths)
            outputs_dec = self.inference_softmax(outputs)
        return outputs_dec

    def compute_evaluation_metrics(self, outputs, labels):
        out_true = self.validation_decoder.label_decoding(labels)
        out_pred = self.validation_decoder.greedy_output_decoding(outputs)
        batch_wer, wer_ref_lens = MCVmetrics.batch_wer(out_true, out_pred, ignore_case=True)
        batch_cer, cer_ref_lens = MCVmetrics.batch_cer(out_true, out_pred, ignore_case=True)
        batch_wavg_wer = MCVmetrics.weighted_avg_wer(batch_wer, wer_ref_lens)
        batch_wavg_cer = MCVmetrics.weighted_avg_cer(batch_cer, cer_ref_lens)
        return batch_wavg_wer, batch_wavg_cer

    def beam_decoding(self, outputs):
        beam_results, _, _, out_lens = self.beam_decoder.decode(outputs)
        beams = [beam_results[b][0][:out_lens[b][0]].tolist() for b in range(outputs.shape[0])]
        output_str = self.validation_decoder.beam_decoding(beams, blank_label=self.blank_label)
        return output_str
    
    def greedy_decoding(self, outputs):
        return self.validation_decoder.greedy_output_decoding(outputs, blank_label=self.blank_label)

    def label_decoding(self, labels):
        return self.validation_decoder.label_decoding(labels, blank_label=self.blank_label)

    def load_saved_model(self):
        print(f'Loading model parameters from: {self.model_load_path}\n')
        return self.load_state_dict(state_dict=torch.load(self.model_load_path))

    def save_model(self):
        print(f'Saving model parameters to: {self.model_save_path}\n')
        torch.save(self.state_dict(),self.model_save_path)
