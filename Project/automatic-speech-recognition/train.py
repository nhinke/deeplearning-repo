# Nick Hinke and Katie Brandegee
# 520.638 Deep Learning
# Final Project - Automatic Speech Recognition
#
# 
#

import os
import csv
import time
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
from model2 import DeepSpeech
from model1 import TorchDeepSpeech

import kenlm
from ctcdecode import CTCBeamDecoder

def plot_spectrogram(spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)

def get_training_logs(dir_path, csv_header, train_file='training_loss.csv', valid_file='validation_loss.csv'):

    train_full = dir_path + '/' + train_file
    valid_full = dir_path + '/' + valid_file
    logs_existed = True

    if (not os.path.isdir(dir_path)):
        os.makedirs(dir_path)
        logs_existed = False

    if (not os.path.isfile(train_full)):
        with open(train_full,'w',newline='') as log_train:
            log_writer = csv.writer(log_train)
            log_writer.writerow(csv_header)
            log_train.close()
            logs_existed = False

    if (not os.path.isfile(valid_full)):
        with open(valid_full,'w',newline='') as log_valid:
            log_writer = csv.writer(log_valid)
            log_writer.writerow(csv_header)
            log_valid.close()
            logs_existed = False

    return logs_existed, train_full, valid_full

def get_current_training_seq_and_epoch(logs_already_existed, training_log_file):

    if (logs_already_existed):
        with open(training_log_file, 'r', newline='') as log_train:
            final_line = log_train.readlines()[-1]
            log_train.close()
        prev_seq = final_line.split(',')[0]
        prev_epoch = final_line.split(',')[1]
        curr_seq = [int(prev_seq)+1 if str(prev_seq) != 'Seq' else 0][0]
        curr_epoch = [int(prev_epoch)+1 if str(prev_epoch) != 'Epoch' else 0][0]
    else:
        curr_seq = 0
        curr_epoch = 0

    return curr_seq, curr_epoch

def write_loss_to_log(file, header, seq, epoch, batch_size, data_size, iter, loss, wer, cer):
    with open(file,'a',newline='') as log:
        log_writer = csv.DictWriter(log,fieldnames=header)
        log_dict = {'Seq':seq, 'Epoch':epoch, 'BatchSize':batch_size, 'DatasetSize':data_size, 'Iteration':iter, 'CTCLoss':loss, 'AvgWER':wer, 'AvgCER':cer}
        log_writer.writerow(log_dict)
        log.close()

def transform_log_iterations_sequentially(train_log, valid_log):
    df_train = pd.read_csv(train_log)
    df_valid = pd.read_csv(valid_log)
    train_iter = np.zeros(shape=(len(df_train),))
    valid_iter = np.zeros(shape=(len(df_valid),))
    raw_train_iter = df_train['Iteration'].tolist()
    raw_valid_iter = df_valid['Iteration'].tolist()
    train_iter[0] = raw_train_iter[0]
    valid_iter[0] = raw_valid_iter[0]
    for i in range(1,train_iter.shape[0]):
        train_iter[i] = train_iter[i-1] + raw_train_iter[i]
    for j in range(1,valid_iter.shape[0]):
        valid_iter[j] = valid_iter[j-1] + raw_valid_iter[j]
    return train_iter, valid_iter

def plot_losses_from_logs(train_log, valid_log):
    df_train = pd.read_csv(train_log)
    df_valid = pd.read_csv(valid_log)
    train_iter, valid_iter = transform_log_iterations_sequentially(train_log, valid_log)
    # train_iter = np.round(np.array(df_train['Epoch']*df_train['DatasetSize']/df_train['BatchSize']+df_train['Iteration']))
    # valid_iter = np.round(np.array(df_valid['Epoch']*df_valid['DatasetSize']/df_valid['BatchSize']+df_valid['Iteration']))
    train_loss = np.array(df_train['CTCLoss'])
    valid_loss = np.array(df_valid['CTCLoss'])
    plt.figure()
    plt.plot(train_iter,train_loss,valid_iter,valid_loss)
    plt.title('CTC Loss')
    plt.xlabel('Iteration')
    plt.legend(['training','validation'])

def plot_metrics_from_logs(train_log, valid_log, on_same_plot=False):
    df_train = pd.read_csv(train_log)
    df_valid = pd.read_csv(valid_log)
    train_iter, valid_iter = transform_log_iterations_sequentially(train_log, valid_log)
    # train_iter = np.round(np.array(df_train['Epoch']*df_train['DatasetSize']/df_train['BatchSize']+df_train['Iteration']))
    # valid_iter = np.round(np.array(df_valid['Epoch']*df_valid['DatasetSize']/df_valid['BatchSize']+df_valid['Iteration']))
    train_wer = np.array(df_train['AvgWER'])
    valid_wer = np.array(df_valid['AvgWER'])
    train_cer = np.array(df_train['AvgCER'])
    valid_cer = np.array(df_valid['AvgCER'])
    if (on_same_plot):
        plt.figure()
        plt.plot(train_iter, train_cer, train_iter, train_wer, valid_iter, valid_cer, valid_iter, valid_wer)
        plt.title('Evaluation Metrics')
        plt.xlabel('Iteration')
        plt.legend(['training CER','training WER','validation CER','validation WER'])
    else:
        plt.figure()
        plt.plot(train_iter, train_cer, valid_iter, valid_cer)
        plt.title('Avg CER')
        plt.xlabel('Iteration')
        plt.legend(['training','validation'])
        plt.figure()
        plt.plot(train_iter, train_wer, valid_iter, valid_wer)
        plt.title('Avg WER')
        plt.xlabel('Iteration')
        plt.legend(['training','validation'])


def main():

    num_epochs = 10
    valid_batch_size = 12
    train_batch_size = 12
    use_my_own_data = False
    perform_validation = True
    # percent_of_validation_set_to_use = 0.10
    percent_of_validation_set_to_use = 0.15
    # percent_of_validation_set_to_use = 0.25

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice to be used during training: {device}')

    train_log_file = 'training_loss.csv'
    valid_log_file = 'validation_loss.csv'

    logs_header = ['Seq','Epoch','BatchSize','DatasetSize','Iteration','CTCLoss','AvgWER','AvgCER']
    logs_path = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Project/automatic-speech-recognition/log'
    logs_existed, log_training_loss, log_validation_loss = get_training_logs(logs_path, logs_header, train_file=train_log_file, valid_file=valid_log_file)
    curr_training_seq, starting_epoch = get_current_training_seq_and_epoch(logs_existed, log_training_loss)

    plot_logs = False
    if (plot_logs):
        plot_losses_from_logs(log_training_loss, log_validation_loss)
        plot_metrics_from_logs(log_training_loss, log_validation_loss, on_same_plot=False)
        plt.show()
        quit()

    # data_path = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Project/automatic-speech-recognition/data/en-1.0'
    # ds_train = MozillaCommonVoiceDataset(root=data_path, tsv='train.tsv', augment=True, resample_audio=False)
    # ds_valid = MozillaCommonVoiceDataset(root=data_path, tsv='dev.tsv', augment=False, resample_audio=False)
    data_path = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Project/automatic-speech-recognition/data/en-5.1'
    ds_train = MozillaCommonVoiceDataset(root=data_path, tsv='train.tsv', augment=True, resample_audio=True)
    ds_valid = MozillaCommonVoiceDataset(root=data_path, tsv='dev.tsv', augment=False, resample_audio=True)
    ds_train_len = ds_train.__len__()
    ds_valid_len = ds_valid.__len__()

    my_data_path = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Project/automatic-speech-recognition/data/my-data'
    ds_mine = myMozillaCommonVoiceDataset(root=my_data_path, csvfile='my-recorded-data.csv', augment=True, resample_audio=True)
    mine_loader = MozillaCommonVoiceDataLoader(ds_mine, batch_size=train_batch_size, shuffle=True, num_workers=4)
    # print(ds_mine.__len__())

    train_loader = MozillaCommonVoiceDataLoader(ds_train, batch_size=train_batch_size, shuffle=True, num_workers=4)
    valid_loader = MozillaCommonVoiceDataLoader(ds_valid, batch_size=valid_batch_size, shuffle=False, num_workers=4)

    # model = ASR()
    # model.load_saved_model()

    model = ASR(load_path_ext='asr-report.pth', save_path_ext='asr-report.pth')
    model.load_saved_model()

    model.to(device)

    print('ASR Model Summary:')
    summary(model)
    print()
    
    model.train()

    print_examples = False
    if (print_examples):
        model.eval()
        for id,data in enumerate(valid_loader): 
            st = time.time()
            inputs, labels, input_lengths, label_lengths = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # input_lengths = input_lengths.to(device)
            # label_lengths = label_lengths.to(device)
            # outputs, loss = model.validation_step(inputs, labels, input_lengths, label_lengths)
            output_probs = model.inference_step(inputs, input_lengths)
            pred_output_str = model.greedy_decoding(output_probs)
            beam_output_str = model.beam_decoding(output_probs)
            true_output_str = model.label_decoding(labels)
            et = time.time()
            for pred1, pred2, true in zip(pred_output_str, beam_output_str, true_output_str):
                print(f"\nActual label:       '{true}'")
                print(f"Greedy prediction:  '{pred1}'")
                print(f"CTCbeam prediction: '{pred2}'")
            print(f'\nTime elapsed when processing batch: {round(et-st,2)} seconds\n')
            break
        quit()


    num_validation_batches = round(1.0*ds_valid_len*percent_of_validation_set_to_use/valid_batch_size)

    if (use_my_own_data):
        train_loader = mine_loader
        perform_validation = False

    epoch_times = list()
    start_time = time.time()
    for epoch in range(num_epochs):
        CTCloss = 0.0
        running_loss = 0.0
        start_ep_time = time.time()
        for id,data in enumerate(train_loader):
            # load batch of image pairs and labels and move to GPU as appropriate
            inputs, labels, input_lengths, label_lengths = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # input_lengths = input_lengths.to(device)
            # label_lengths = label_lengths.to(device)

            outputs, loss = model.training_step(inputs, labels, input_lengths, label_lengths)

            running_loss += loss.item()
            if (id % 100 == 99):   
                CTCloss = running_loss/100.0
                batch_avg_wer, batch_avg_cer = model.compute_evaluation_metrics(outputs, labels)
                print(f'[{epoch+1},{id+1}] Training:   ( CTC loss = {CTCloss:.3f} , WER = {batch_avg_wer:.3f} , CER = {batch_avg_cer:.3f} )')
                write_loss_to_log(log_training_loss, logs_header, curr_training_seq, starting_epoch+epoch, train_batch_size, ds_train_len, id, CTCloss, batch_avg_wer, batch_avg_cer)
                running_loss = 0.0

                if (perform_validation):
                    model.eval()
                    running_loss = 0.0
                    avg_wer_list = list()
                    avg_cer_list = list()
                    # validation_losses = list()
                    for id_val,data in enumerate(valid_loader): 
                        inputs, labels, input_lengths, label_lengths = data
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        # input_lengths = input_lengths.to(device)
                        # label_lengths = label_lengths.to(device)

                        outputs, loss = model.validation_step(inputs, labels, input_lengths, label_lengths)

                        running_loss += loss.item()

                        batch_avg_wer, batch_avg_cer = model.compute_evaluation_metrics(outputs, labels)
                        avg_wer_list.append(batch_avg_wer)
                        avg_cer_list.append(batch_avg_cer)

                        if (id_val >= num_validation_batches):
                            break

                    val_avg_wer = np.mean(avg_wer_list)
                    val_avg_cer = np.mean(avg_cer_list)
                    CTCloss = running_loss/(1.0*(num_validation_batches-1.0))
                    print(f'[{epoch+1},{id+1}] Validation: ( CTC loss = {CTCloss:.3f} , WER = {val_avg_wer:.3f} , CER = {val_avg_cer:.3f} )')
                    write_loss_to_log(log_validation_loss, logs_header, curr_training_seq, starting_epoch+epoch, valid_batch_size, ds_valid_len, id, CTCloss, val_avg_wer, val_avg_cer)
                    running_loss = 0.0

                    model.train()
                    model.scheduler.step(CTCloss)

        epoch_times.append(time.time()-start_ep_time)
        print(f'EPOCH {epoch+1}/{num_epochs} COMPLETE (took {epoch_times[-1]:.1f} seconds)...\n')

    print(f'Total elapsed time: {time.time()-start_time:.1f} seconds')
    model.save_model()



if __name__ == "__main__":
    main()
