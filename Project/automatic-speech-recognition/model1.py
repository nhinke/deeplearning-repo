# based on torchaudio.models.DeepSpeech

import torch
import torchaudio
import MCVmetrics
import MCVencoding

import kenlm
import ctcdecode

class TorchDeepSpeech(torchaudio.models.DeepSpeech):

    kenlm_path = None

    adam_learning_rate = 2.0e-6
    adam_betas = (0.9, 0.999)
    adam_eps = 1e-8
    adam_weight_decay = 1e-5
    adam_learning_anneal = 0.99
    abs_path = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Project/automatic-speech-recognition/model-params/'

    def __init__(self, num_features=64, num_classes=28, blank_label=27, load_path_ext='asr-torchDS.pth',save_path_ext='asr-torchDS.pth'):
        self.num_classes = num_classes
        self.num_features = num_features
        self.blank_label = blank_label
        self.model_load_path = self.abs_path+load_path_ext
        self.model_save_path = self.abs_path+save_path_ext
        super(TorchDeepSpeech,self).__init__(n_feature=self.num_features,n_class=self.num_classes)
        self.validation_decoder = MCVencoding.EncoderDecoder()
        self.beam_decoder = ctcdecode.CTCBeamDecoder(labels=list("-abcdefghijklmnopqrstuvwxyz_"), blank_id=self.blank_label, log_probs_input=False, model_path=self.kenlm_path)
        self.criterion = torch.nn.CTCLoss(blank=self.blank_label, reduction='mean', zero_infinity=True)
        # self.optimizer = torch.optim.SGD(params=self.parameters(), lr=1e-3, momentum=0.9)
        self.optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.adam_learning_rate, betas=self.adam_betas, eps=self.adam_eps, weight_decay=self.adam_weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=self.adam_learning_anneal)

    def training_step(self, inputs, labels, input_lengths, label_lengths):
        inputs = inputs.transpose(2,3)
        self.optimizer.zero_grad()
        outputs = self(inputs)
        loss = self.criterion(outputs.transpose(0,1), labels, input_lengths, label_lengths)
        loss.backward()
        self.optimizer.step()
        return outputs, loss

    def validation_step(self, inputs, labels, input_lengths, label_lengths):
        with torch.no_grad():
            inputs = inputs.transpose(2,3)
            outputs = self(inputs)
            loss = self.criterion(outputs.transpose(0,1), labels, input_lengths, label_lengths)
        return outputs, loss

    def inference_step(self, inputs, input_lengths):
        with torch.no_grad():
            inputs = inputs.transpose(2,3)
            outputs = self(inputs)
        return outputs

    def beam_decoding(self, outputs):
        beam_results, _, _, out_lens = self.beam_decoder.decode(outputs)
        beams = [beam_results[b][0][:out_lens[b][0]].tolist() for b in range(outputs.shape[0])]
        output_str = self.validation_decoder.beam_decoding(beams, blank_label=self.blank_label)
        return output_str
    
    def greedy_decoding(self, outputs):
        return self.validation_decoder.greedy_output_decoding(outputs, blank_label=self.blank_label)

    def label_decoding(self, labels):
        return self.validation_decoder.label_decoding(labels, blank_label=self.blank_label)

    def compute_evaluation_metrics(self, outputs, labels):
        out_pred = self.validation_decoder.greedy_output_decoding(outputs)
        out_true = self.validation_decoder.label_decoding(labels)
        batch_wer, wer_ref_lens = MCVmetrics.batch_wer(out_true, out_pred, ignore_case=True)
        batch_cer, cer_ref_lens = MCVmetrics.batch_cer(out_true, out_pred, ignore_case=True)
        batch_wavg_wer = MCVmetrics.weighted_avg_wer(batch_wer, wer_ref_lens)
        batch_wavg_cer = MCVmetrics.weighted_avg_cer(batch_cer, cer_ref_lens)
        return batch_wavg_wer, batch_wavg_cer

    def load_saved_model(self):
        print(f'Loading model parameters from: {self.model_load_path}\n')
        return self.load_state_dict(state_dict=torch.load(self.model_load_path))

    def save_model(self):
        print(f'Saving model parameters to: {self.model_save_path}\n')
        torch.save(self.state_dict(),self.model_save_path)
