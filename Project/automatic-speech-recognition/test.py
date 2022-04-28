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

    # for plotting: current iteration = epoch*(dataset_size/batch_size)+iter
    #   since (dataset_size/batch_size) = iter per epoch

    num_epochs = 1
    valid_batch_size = 8
    train_batch_size = 12
    perform_validation = True

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice to be used during training: {device}')

    logs_header = ['Seq','Epoch','BatchSize','DatasetSize','Iteration','CTCLoss','AvgWER','AvgCER']
    logs_path = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Project/automatic-speech-recognition/log'
    logs_existed, log_training_loss, log_validation_loss = get_training_logs(logs_path,logs_header)
    curr_training_seq, starting_epoch = get_current_training_seq_and_epoch(logs_existed,log_training_loss)

    plot_logs = False
    if (plot_logs):
        plot_losses_from_logs(log_training_loss,log_validation_loss)
        plot_metrics_from_logs(log_training_loss,log_validation_loss,on_same_plot=False)
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
    # print(ds_train_len)
    # print(ds_valid_len)

    train_loader = MozillaCommonVoiceDataLoader(ds_train, batch_size=train_batch_size, shuffle=True, num_workers=4)
    valid_loader = MozillaCommonVoiceDataLoader(ds_valid, batch_size=valid_batch_size, shuffle=True, num_workers=4)

    model = ASR()
    # model = ASR(load_path_ext='asr-ten.pth',save_path_ext='asr-ten.pth') # overfit to ten batches
    model.load_saved_model()
    model.to(device)
    model.train()

    print_examples = True
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




    epoch_times = list()
    start_time = time.time()
    for epoch in range(num_epochs):
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
                print(f'[{epoch+1},{id+1}] Training: ( CTC loss = {CTCloss:.3f} , WER = {batch_avg_wer:.3f} , CER = {batch_avg_cer:.3f} )')
                write_loss_to_log(log_training_loss, logs_header, curr_training_seq, starting_epoch+epoch, train_batch_size, ds_train_len, id, CTCloss, batch_avg_wer, batch_avg_cer)
                running_loss = 0.0

        if (perform_validation):
            model.eval()
            running_loss = 0.0
            validation_losses = list()
            for id,data in enumerate(valid_loader): 
                inputs, labels, input_lengths, label_lengths = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                # input_lengths = input_lengths.to(device)
                # label_lengths = label_lengths.to(device)

                outputs, loss = model.validation_step(inputs, labels, input_lengths, label_lengths)

                running_loss += loss.item()
                if (id % 100 == 99):   
                    CTCloss = running_loss/100.0
                    validation_losses.append(CTCloss)
                    batch_avg_wer, batch_avg_cer = model.compute_evaluation_metrics(outputs, labels)
                    print(f'[{epoch+1},{id+1}] Validation: ( CTC loss = {CTCloss:.3f} , WER = {batch_avg_wer:.3f} , CER = {batch_avg_cer:.3f} )')
                    write_loss_to_log(log_validation_loss, logs_header, curr_training_seq, starting_epoch+epoch, valid_batch_size, ds_valid_len, id, CTCloss, batch_avg_wer, batch_avg_cer)
                    running_loss = 0.0
            model.train()
            avg_validation_loss = float(np.mean(validation_losses))
            model.scheduler.step(avg_validation_loss)

        epoch_times.append(time.time()-start_ep_time)
        print(f'EPOCH {epoch+1}/{num_epochs} COMPLETE (took {epoch_times[-1]:.1f} seconds)...\n')

    print(f'Total elapsed time: {time.time()-start_time:.1f} seconds')
    model.save_model()

    quit()

    data_path = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Project/automatic-speech-recognition/data/en'
    ds_train = MozillaCommonVoiceDataset(root=data_path, tsv='train.tsv', augment=True, resample_audio=False)
    ds_valid = MozillaCommonVoiceDataset(root=data_path, tsv='dev.tsv', augment=False, resample_audio=False)
    # ds_test = MozillaCommonVoiceDataset(root=data_path, tsv='test.tsv', augment=False, resample_audio=False)
    # print(f'len train: {ds_train.__len__()}')
    # print(f'len valid: {ds_valid.__len__()}')
    # print(f'len test:  {ds_test.__len__()}')

    # CTCloss=123
    # iteration=0
    # epoch=0
    # avg_wer=1.0
    # avg_cer=0.5
    # write_loss_to_log(log_training_loss,logs_header,curr_training_seq,starting_epoch+epoch,iteration,CTCloss,avg_wer,avg_cer)
    # write_loss_to_log(log_training_loss,logs_header,curr_training_seq,starting_epoch+epoch,iteration,CTCloss,avg_wer,avg_cer)
    # write_loss_to_log(log_validation_loss,logs_header,curr_training_seq,starting_epoch+epoch,iteration,CTCloss,avg_wer,avg_cer)
    # write_loss_to_log(log_validation_loss,logs_header,curr_training_seq,starting_epoch+epoch,iteration,CTCloss,avg_wer,avg_cer)


    # print(curr_training_seq)
    # print(starting_epoch)

    # df_train = pd.read_csv(log_training_loss)
    # df_valid = pd.read_csv(log_validation_loss)

    # quit()



    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)



    # model = ASR()
    # model2 = torchaudio.models.DeepSpeech(n_feature=64,n_class=28)

    # model_load_path = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Project/automatic-speech-recognition/model/asr3-deepspeech.pth'
    # model_save_path ='/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Project/automatic-speech-recognition/model/asr3-deepspeech.pth'
    # model.load_state_dict(torch.load(model_load_path))

    batch_size = 12
    epochs = 100

    train_loader = MozillaCommonVoiceDataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = MozillaCommonVoiceDataLoader(ds_valid, batch_size=8, shuffle=False, num_workers=4)
    test_loader = MozillaCommonVoiceDataLoader(ds_test, batch_size=2, shuffle=False, num_workers=4)

    metric_evluation = True
    perform_validation = False

    # m = ASR()
    m = ASR(load_path_ext='asr-single.pth',save_path_ext='asr-single.pth') # overfit to one batch
    # m = ASR(load_path_ext='asr-single.pth',save_path_ext='asr-ten.pth') # overfit to ten batches

    # m = DeepSpeech(load_path_ext='asr3-ds.pth',save_path_ext='asr-DS-test.pth')
    # m = TorchDeepSpeech(load_path_ext='asr-torchDS-test.pth',save_path_ext='asr-torchDS-test.pth')
    # m = TorchDeepSpeech(load_path_ext='asr3-ds.pth') # best version but only trained on one batch (set ds to only one batch)

    m.load_saved_model()
    m.to(device)

    if (metric_evluation):
        m.eval()
        start = time.time()
        epoch_times = list()
        for epoch in range(1):
            start_ep = time.time()
            train_loss = 0.0
            valid_loss = list()
            running_loss = 0.0
            for id,data in enumerate(train_loader): 
                inputs, labels, input_lengths, label_lengths = data
                inputs = inputs.to(device)
                labels.to(device)
                input_lengths.to(device)
                label_lengths.to(device)
                outputs_dec,loss = m.validation_step(inputs,labels,input_lengths,label_lengths)
                running_loss += loss.item()
                out_pred = m.validation_decoder.greedy_output_decoding(outputs_dec)
                out_true = m.validation_decoder.label_decoding(labels)
                batch_wer,wer_ref_lens = metrics.batch_wer(out_true, out_pred, ignore_case=True)
                batch_cer,cer_ref_lens = metrics.batch_cer(out_true, out_pred, ignore_case=True)
                avg_wer = round(metrics.weighted_avg_wer(batch_wer,wer_ref_lens),4)
                avg_cer = round(metrics.weighted_avg_cer(batch_cer,cer_ref_lens),4)
                if (id > 5):
                    break
        wer_list2 = [round(wer,2) for wer in batch_wer]
        cer_list2 = [round(cer,2) for cer in batch_cer]
        print()
        print(f'WER: {wer_list2}')
        print(f'CER: {cer_list2}')
        print()
        print(f'avg_wer: {avg_wer}')
        print(f'avg_cer: {avg_cer}')
        print()
        print(f'Prediction:   {out_pred}')
        print(f'Actual label: {out_true}')
        print()
        plt.show()
        quit()


    m.train()
    start = time.time()
    epoch_times = list()
    for epoch in range(epochs):
        start_ep = time.time()
        train_loss = 0.0
        valid_loss = list()
        running_loss = 0.0
        for id,data in enumerate(train_loader):
            # load batch of image pairs and labels and move to GPU as appropriate
            inputs, labels, input_lengths, label_lengths = data
            inputs = inputs.to(device)
            labels.to(device)
            input_lengths.to(device)
            label_lengths.to(device)

            outputs_dec,loss = m.training_step(inputs,labels,input_lengths,label_lengths)

            train_loss += loss.item()
            print(train_loss) # for overfitting to single batch

            if (id % 100 == 99):   
                print(f'[{epoch+1},{id+1}] training loss: {train_loss/100.0:.4f}')
                train_loss = 0.0
                # print(m.validation_decoder.greedy_output_decoding(outputs_dec))
                # print(m.validation_decoder.label_decoding(labels))
        if (perform_validation):
            for id,data in enumerate(valid_loader): 
                inputs, labels, input_lengths, label_lengths = data
                inputs = inputs.to(device)
                labels.to(device)
                input_lengths.to(device)
                label_lengths.to(device)
                outputs_dec_val,loss = m.validation_step(inputs,labels,input_lengths,label_lengths)
                running_loss += loss.item()
                if (id % 100 == 99):   
                    print(f'[{epoch+1},{id+1}] valid loss: {running_loss/100.0:.3f}')
                    valid_loss.append(running_loss)
                    running_loss = 0.0
                if (len(valid_loss) > 2):
                    break

        # avg_loss = np.mean(np.array(valid_loss))
        # scheduler.step(avg_loss)
        epoch_times.append(time.time()-start_ep)

        # print(outputs_dec.shape)
        print()
        print(m.validation_decoder.greedy_output_decoding(outputs_dec))
        # print(m.validation_decoder.label_decoding(labels))
        print()
        print(f'EPOCH {epoch+1}/{epochs} COMPLETE (took {epoch_times[-1]:.1f} seconds)...\n')

    print([round(t,1) for t in epoch_times])
    print(f'total elapsed time: {time.time()-start:.1f} seconds')


    m.save_model()
    
    quit()
    

    MODEL2 = False
    if MODEL2:
        model2.to(device)
        model2.train()
        # for param in model2.parameters():
        #     param.requires_grad = True 
        # epochs = 20
        start = time.time()
        epoch_times = list()
        for epoch in range(epochs):
            start_ep = time.time()
            train_loss = 0.0
            valid_loss = list()
            running_loss = 0.0
            for id,data in enumerate(train_loader):
                # load batch of image pairs and labels and move to GPU as appropriate
                # print(data)
                inputs, labels, input_lengths, label_lengths = data
                inputs = inputs.transpose(2,3)
                print(inputs.shape)
                inputs = inputs.to(device)
                labels.to(device)
                input_lengths.to(device)
                label_lengths.to(device)
                optimizer2.zero_grad()
                outputs = model2(inputs)
                # outputs_dec = torch.nn.functional.softmax(outputs,dim=-1)
                # print(outputs.shape)
                # print(labels.shape)
                # print(input_lengths.shape)
                # print(label_lengths.shape)
                loss = criterion(outputs.transpose(0,1), labels, input_lengths, label_lengths)
                loss.backward()
                optimizer2.step()
                train_loss += loss.item()
                print(train_loss)
            if (epoch % 5 == 4):
                print(epoch)
                print(decoder.greedy_output_decoding(outputs))
        print(f'Finished training!\nSaving model parameters to: {model_save_path}\n')
        torch.save(model2.state_dict(),model_save_path)
        quit()


    model.to(device)
    model.train()

    TEST = False
    if TEST:
        model.eval()
        with torch.no_grad():
            for id,data in enumerate(test_loader):
                inputs, labels, input_lengths, label_lengths = data
                inputs = inputs.to(device)
                labels.to(device)
                input_lengths.to(device)
                label_lengths.to(device)
                outputs,_ = model(inputs)

                # print(outputs.type())

                out_str = decoder.greedy_output_decoding(outputs)
                label_str = decoder.label_decoding(labels)
                print(out_str)
                print(label_str)
                quit()




    start = time.time()
    epoch_times = list()
    for epoch in range(epochs):
        start_ep = time.time()
        train_loss = 0.0
        valid_loss = list()
        running_loss = 0.0
        for id,data in enumerate(train_loader):
            # load batch of image pairs and labels and move to GPU as appropriate
            # print(data)
            inputs, labels, input_lengths, label_lengths = data
            inputs = inputs.to(device)
            labels.to(device)
            input_lengths.to(device)
            label_lengths.to(device)

            # print(inputs.type())
            # print(inputs.shape)
            # print(labels.shape)
            # print(input_lengths)
            # print(label_lengths)
            optimizer.zero_grad()
            # outputs,_ = model(inputs)
            outputs,seq_lengths = model(inputs,input_lengths)

            # print(outputs)
            outputs_dec = torch.nn.functional.softmax(outputs,dim=-1)
            outputs = outputs.transpose(0,1)
            outputs = outputs.log_softmax(-1)
            # print(outputs.shape)
            # print(outputs.shape)
            # print(input_lengths)
            # print(seq_lengths)
            # print(outputs.shape)
            # print(outputs_dec[0].shape)
            # loss = criterion(outputs.transpose(0,1), labels, seq_lengths, label_lengths)
            loss = criterion(outputs, labels, seq_lengths, label_lengths)
            loss.backward()
            optimizer.step()

            # optimizer.step()
            train_loss += loss.item()
            print(train_loss)

            if (id % 100 == 99):   
                print(f'[{epoch+1},{id+1}] training loss: {train_loss/100.0:.3f}')
                train_loss = 0.0
                print(decoder.greedy_output_decoding(outputs))
                print(decoder.label_decoding(labels))
                # break
        # for id,data in enumerate(valid_loader):
        #     inputs, labels, input_lengths, label_lengths = data
        #     inputs = inputs.to(device)
        #     labels.to(device)
        #     input_lengths.to(device)
        #     label_lengths.to(device)

        #     optimizer.zero_grad()
        #     outputs,h = model(inputs)
        #     # print(outputs.shape)
        #     loss = criterion(outputs.transpose(0,1), labels, input_lengths, label_lengths)
        #     running_loss += loss.item()
        #     if (id % 100 == 99):   
        #         print(f'[{epoch+1},{id+1}] valid loss: {running_loss/100.0:.3f}')
        #         valid_loss.append(running_loss)
        #         running_loss = 0.0
        #         # break
        #     if (len(valid_loss) > 2):
        #         break

        # avg_loss = np.mean(np.array(valid_loss))
        # scheduler.step(avg_loss)
        epoch_times.append(time.time()-start_ep)
        print(f'EPOCH {epoch+1}/{epochs} COMPLETE (took {epoch_times[-1]:.1f} seconds)...\n')

        if (epoch % 5 == 4):
            print(decoder.greedy_output_decoding(outputs_dec))
            # print(decoder.label_decoding(labels))

    print([round(t,1) for t in epoch_times])
    print(f'total elapsed time: {time.time()-start:.1f} seconds')

    # model_path = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Project/automatic-speech-recognition/model/asr.pth'

    print(f'Finished training!\nSaving model parameters to: {model_save_path}\n')
    torch.save(model.state_dict(),model_save_path)
    
    quit()

    print(spec.shape)
    print(spec.reshape(1,spec.shape[0],spec.shape[1],spec.shape[2]).shape)
    spec = spec.reshape(1,spec.shape[0],spec.shape[1],spec.shape[2])
    model(spec)

    quit()


    count = 0
    max = 6.0
    min = 2.0
    for ind,sample in enumerate(ds):
        sig,sr,data = sample
        sig_time = (sig.shape[1]*1000.0/sr)/1000.0
        if (sig_time <= max and sig_time >= min):
            count += 1
    print(count)
    quit()



    ind = 0
    sig,data = ds_train.__getitem__(ind)
    # print(sig.shape)
    # plt.figure()
    # plt.imshow(sig[0])
    # plt.show()

    # spec1 = transforms.AmplitudeToDB(top_db=80)(transforms.MelSpectrogram(sr,n_fft=1024,n_mels=64,hop_length=None)(sig))

    
    # spec1 = transforms.AmplitudeToDB(top_db=80)(transforms.MelSpectrogram(sr,n_fft=1024,n_mels=64,hop_length=None)(sig))
    # # print(spec.shape)
    # spec2 = transforms.AmplitudeToDB(top_db=80)(transforms.MelSpectrogram(sr,n_fft=400,n_mels=64,hop_length=None)(sig))
    # spec3 = transforms.AmplitudeToDB(top_db=80)(transforms.MelSpectrogram(sr,n_fft=1024,n_mels=128,hop_length=None)(sig))
    # spec4 = transforms.AmplitudeToDB(top_db=80)(transforms.MelSpectrogram(sr,n_fft=400,n_mels=128,hop_length=None)(sig))
    # spec5 = transforms.MFCC(sr,n_mfcc=64,melkwargs={'n_fft':1024,'n_mels':128})(sig)
    # spec6 = transforms.MFCC(sr,n_mfcc=40,melkwargs={'n_fft':1024,'n_mels':64})(sig)
    # print(f'spec1.shape: {spec1.shape}')
    # print(f'spec2.shape: {spec2.shape}')
    # print(f'spec3.shape: {spec3.shape}')
    # print(f'spec4.shape: {spec4.shape}')
    # print(f'spec5.shape: {spec5.shape}')
    # print(f'spec6.shape: {spec6.shape}')

    # plt.figure()
    # plt.imshow(spec1[0])
    # plt.title('n_fft = 1024 , n_mels = 64')
    # # plt.figure()
    # # plt.imshow(spec2[0])
    # # plt.title('n_fft = 400 , n_mels = 64')
    # plt.figure()
    # plt.imshow(spec3[0])
    # plt.title('n_fft = 1024 , n_mels = 128')
    # # plt.figure()
    # # plt.imshow(spec4[0])
    # # plt.title('n_fft = 400 , n_mels = 128')
    # plt.figure()
    # plt.imshow(spec5[0])
    # plt.title('n_fft = 1024 , n_mels = 128, n_mfcc = 40')
    # plt.figure()
    # plt.imshow(spec6[0])
    # plt.title('n_fft = 400 , n_mels = 128, n_mfcc = 40')

    # plt.show()
    # quit()



    i = 0
    max_len = -1.0
    min_len = 99999.0
    for i,sample in enumerate(ds_train):
        # ind = 1
        # print(ds.__getitem__(ind))

        # if (i>10):
        #     quit()

        # print(f'\nind: {i}')
        # print(f'\nind: {ind}')
        # sample = ds.__getitem__(ind)

        # print(out)
        # print(out.shape)
        sig,data = sample
        # print(sr)

        # print(f'sr: {sr}')
        # print(f'sig.shape: {sig.shape}')

        sig_time = (sig.shape[1]*1000.0/sr)/1000.0

        if (sig.shape[0] != 1):
            print(f'num_channels: {sig.shape[0]} at index {i}')

        if (sig_time > max_len):
            max_len = sig_time
            print(f'new max length of {max_len:.2f} seconds at index {i}')
        elif (sig_time < min_len):
            min_len = sig_time
            print(f'new min length of {min_len:.2f} seconds at index {i}')

        if (sr != 48000):
            print(f'sample_Rate: {sr} at index {i}')

        
        # print(sig_time)

        n_mels=64
        n_fft=1024
        top_db=80
        hop_len=None
        # sig,sr = aud1
        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc and convert to decibels
        spec1 = torchaudio.transforms.MelSpectrogram(sr,n_fft=n_fft,n_mels=n_mels,hop_length=hop_len)(sig)
        spec1 = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(spec1)

        # spec2 = transforms.AmplitudeToDB()(transforms.MelSpectrogram()(sig))
        # spec2 = transforms.MFCC(sr)(sig)

        # print(sr)
        # print(sig.shape)
        # print(f'spec.shape: {spec1.shape}')



    plt.figure()
    plt.imshow(spec1[0])
    # plt.figure()
    # plt.imshow(spec2[0])
    plt.show(block=1)
    # print(spec1.shape)

    # plot_spectrogram(spec1[0])
    # plot_spectrogram(spec2[0])

    # model=ASR()
    # model(x)

    # ds = datasets.COMMONVOICE(root='/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Project/automatic-speech-recognition/data/en')
    # print(ds.__len__())



if __name__ == "__main__":
    main()
