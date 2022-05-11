# Nick Hinke and Katie Brandegee
# 520.638 Deep Learning
# Final Project - Automatic Speech Recognition
#
# 
#

import os
import csv
import time
from matplotlib import use
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import librosa
import torchaudio
from torchinfo import summary
from MCVencoding import EncoderDecoder
from myMCVdataset import myMozillaCommonVoiceDataset
from MCVdataset import MozillaCommonVoiceDataset
from MCVdataloader import MozillaCommonVoiceDataLoader

from model3 import ASR

def main():

    valid_batch_size = 12
    train_batch_size = 12
    use_my_own_data = False

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice to be used during training: {device}')

    data_path = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Project/automatic-speech-recognition/data/en-1.0'
    ds_test = MozillaCommonVoiceDataset(root=data_path, tsv='test.tsv', augment=False, resample_audio=False)
    # data_path = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Project/automatic-speech-recognition/data/en-5.1'
    # ds_test = MozillaCommonVoiceDataset(root=data_path, tsv='test.tsv', augment=False, resample_audio=True)
    test_loader = MozillaCommonVoiceDataLoader(ds_test, batch_size=valid_batch_size, shuffle=True, num_workers=4)
    ds_test_len = ds_test.__len__()
    print(f'\nTest dataset length: {ds_test_len}\n')

    my_data_path = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Project/automatic-speech-recognition/data/my-data'
    ds_mine = myMozillaCommonVoiceDataset(root=my_data_path, csvfile='my-recorded-data.csv', augment=True, resample_audio=True)
    mine_loader = MozillaCommonVoiceDataLoader(ds_mine, batch_size=train_batch_size, shuffle=True, num_workers=4)
    # print(ds_mine.__len__())

    model2 = ASR()
    model2.load_saved_model()

    model = ASR(load_path_ext='asr-report.pth', save_path_ext='asr-report.pth')
    model.load_saved_model()

    model.to(device)
    model2.to(device)

    model.eval()
    model2.eval()

    # print('ASR Model Summary:')
    # summary(model)
    # print()

    if (use_my_own_data):
        test_loader = mine_loader

    avg_wer_list = list()
    avg_cer_list = list()
    avg_wer_list2 = list()
    avg_cer_list2 = list()

    for id, data in enumerate(test_loader): 
        inputs, labels, input_lengths, label_lengths = data
        inputs = inputs.to(device)
        # labels = labels.to(device)
        # input_lengths = input_lengths.to(device)
        # label_lengths = label_lengths.to(device)

        outputs = model.inference_step(inputs, input_lengths)
        outputs2 = model2.inference_step(inputs, input_lengths)

        batch_avg_wer, batch_avg_cer = model.compute_evaluation_metrics(outputs, labels)
        avg_wer_list.append(batch_avg_wer)
        avg_cer_list.append(batch_avg_cer)

        batch_avg_wer2, batch_avg_cer2 = model2.compute_evaluation_metrics(outputs2, labels)
        avg_wer_list2.append(batch_avg_wer2)
        avg_cer_list2.append(batch_avg_cer2)

    val_avg_wer = np.mean(avg_wer_list)
    val_avg_cer = np.mean(avg_cer_list)

    val_avg_wer2 = np.mean(avg_wer_list2)
    val_avg_cer2 = np.mean(avg_cer_list2)

    print()
    print(f'Partially-trained model CER:  ( mean = {val_avg_cer2:.2f} , max = {max(avg_cer_list2):.2f} , min = {min(avg_cer_list2):.2f} )')
    print(f'Partially-trained model WER:  ( mean = {val_avg_wer2:.2f} , max = {max(avg_wer_list2):.2f} , min = {min(avg_wer_list2):.2f} )')
    print()
    print(f'Totally-trained model CER:    ( mean = {val_avg_cer:.2f} , max = {max(avg_cer_list):.2f} , min = {min(avg_cer_list):.2f} )')
    print(f'Totally-trained model WER:    ( mean = {val_avg_wer:.2f} , max = {max(avg_wer_list):.2f} , min = {min(avg_wer_list):.2f} )')
    print()


if __name__ == "__main__":
    main()