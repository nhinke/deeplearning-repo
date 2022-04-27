# from numpy import pad
import torch
# import torchaudio
# import MCVdataset

def _collate_fn2(batch):
    # def func(p):
    #     return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(2), reverse=True)
    batch2 = sorted(batch, key=lambda sample: len(sample[1]['label']), reverse=True)
    # print(sample[0].shape)
    # print(len(batch[0]))
    longest_sample = batch[0]
    longest_sample_label = batch2[0]
    freq_size = longest_sample[0].size(1)
    # print(freq_size)
    minibatch_size = len(batch)
    max_seqlength = longest_sample[0].size(2)
    max_label_len = len(longest_sample_label[1]['label'])
    # print(max_seqlength)
    # print(max_label_len)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    targets = torch.zeros(minibatch_size, max_label_len)
    input_sizes2 = torch.IntTensor(minibatch_size)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets2 = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]['label']
        # print()
        # print(tensor[0].shape)
        seq_length = tensor.size(2)
        label_length = len(target)
        # print(inputs[x][0].shape)
        # print(inputs[x][0].narrow(1,0,seq_length).shape)
        # print(inputs[x][0].narrow(0,0,seq_length).shape)
        # print(torch.IntTensor(target).shape)
        # print(targets[x].shape)
        targets[x].narrow(0,0,label_length).copy_(torch.IntTensor(target))
        # print(targets[x])
        inputs[x][0].narrow(1,0,seq_length).copy_(tensor[0])
        input_percentages[x] = seq_length / float(max_seqlength)
        input_sizes2[x] = seq_length
        target_sizes[x] = label_length
        targets2.extend(target)
    targets2 = torch.tensor(targets2, dtype=torch.long)
    input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
    # print(target_sizes)
    # print(input_sizes)
    # print(input_sizes2)
    # print(inputs.shape)
    # print(targets.shape)
    return inputs, targets, input_sizes, target_sizes

def _collate_fn(data):
    '''
    Padds batch of variable length
    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    spectrograms = list()
    labels = list()
    input_lengths = list()
    label_lengths = list()
    for (spectrogram, label_meta) in data:
        if spectrogram is None:
            continue

        # print(label['sentence'])
        # print(list(label['sentence']))
        # print(type(list(label['sentence'])))
        label = list(label_meta['label'])
        input_length = spectrogram.shape[-1]//2 # check that this is right
        input_length = spectrogram.shape[-1]
        label_length = len(label)
        spectrograms.append(spectrogram.squeeze(0).transpose(0, 1))
        labels.append(torch.Tensor(label))
        input_lengths.append(input_length)
        label_lengths.append(label_length)

    spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=27)
    input_lengths = torch.IntTensor(input_lengths)
    label_lengths = torch.IntTensor(label_lengths)
    return spectrograms, labels, input_lengths, label_lengths


class MozillaCommonVoiceDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super(MozillaCommonVoiceDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn2
