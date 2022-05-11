
import csv
import torch
import string
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import LabelEncoder

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, layers=1, dropout=0.1, bidirectional=True):
        super(EncoderRNN, self).__init__()

        self.directions = 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = layers
        self.dropout = dropout
        self.embedder = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                         num_layers=layers, dropout=dropout,
                         bidirectional=bidirectional, batch_first=False)
        self.fc = nn.Linear(hidden_size*self.directions, hidden_size)
        
    def forward(self, input_data, h_hidden, c_hidden):
        embedded_data = self.embedder(input_data)
        embedded_data = self.dropout(embedded_data)
        hiddens, outputs = self.lstm(embedded_data, (h_hidden, c_hidden))
        return hiddens, outputs

    def create_init_hiddens(self, batch_size):
        h_hidden = Variable(torch.zeros(self.num_layers*self.directions, batch_size, self.hidden_size))
        c_hidden = Variable(torch.zeros(self.num_layers*self.directions, batch_size, self.hidden_size))
#         torch.cuda.is_available():
#              h_hidden.cuda(), c_hidden.cuda()
# 		else:
        return h_hidden, c_hidden


class DecoderAttn(nn.Module):
    def __init__(self, hidden_size, output_size, layers=1, dropout=0.1, bidirectional=True):
        super(DecoderAttn, self).__init__()

        if bidirectional:
            self.directions = 2
        else:
            self.directions = 1
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = layers
        self.dropout = dropout
        self.embedder = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.score_learner = nn.Linear(hidden_size*self.directions,
                                 hidden_size*self.directions)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                        num_layers=layers, dropout=dropout,
                        bidirectional=bidirectional, batch_first=False)
        self.context_combiner = nn.Linear((hidden_size*self.directions)
                                      + (hidden_size*self.directions), hidden_size)
        self.tanh = nn.Tanh()
        self.output = nn.Linear(hidden_size, output_size)
        self.soft = nn.Softmax(dim=1)
        self.log_soft = nn.LogSoftmax(dim=1)
        

    def forward(self, input_data, h_hidden, c_hidden, encoder_hiddens):
 
        embedded_data = self.embedder(input_data)
        embedded_data = self.dropout(embedded_data)
        batch_size = embedded_data.shape[1]
        hiddens, outputs = self.lstm(embedded_data, (h_hidden, c_hidden))
        top_hidden = outputs[0].view(self.num_layers, self.directions,
                               hiddens.shape[1],
                               self.hidden_size)[self.num_layers-1]
        top_hidden = top_hidden.permute(1, 2,0).contiguous().view(batch_size,-1, 1)

        prep_scores = self.score_learner(encoder_hiddens.permute(1, 0,2))
        scores = torch.bmm(prep_scores, top_hidden)
        attn_scores = self.soft(scores)
        con_mat = torch.bmm(encoder_hiddens.permute(1, 2,0),attn_scores)
        h_tilde = self.tanh(self.context_combiner(torch.cat((con_mat,
                                                          top_hidden), dim=1)
                                              .view(batch_size, -1)))
        pred = self.output(h_tilde)
        pred = self.log_soft(pred)

        return pred, outputs


def demo(input_string):
      
    path = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Project/neural-machine-translation/'
    
    data = np.genfromtxt(path+'spa.txt',delimiter='\t', dtype = 'str', encoding = "utf8")
    data_x = data[0:80000, 0]
    data_y = data[0:80000, 1]
    numData = len(data_x)
    
    dictx=[]
    for a in range(0, len(data_x)):
        t = data_x[a].split()
        for word in t:
            dictx.append(word)
    
    label_encoderx = LabelEncoder()
    label_encoderx.fit(dictx)
    label_encoderx.classes_ = np.append(label_encoderx.classes_, '<')
    
    x_SOS = len(label_encoderx.classes_)-1
    
    label_encoderx.classes_ = np.append(label_encoderx.classes_, '>')
    x_EOS = len(label_encoderx.classes_)-1


    dicty=[]
    for a in range(0, len(data_y)):
        t = data_y[a].split()
        for word in t:
            dicty.append(word)
    
    label_encodery = LabelEncoder()
    label_encodery.fit(dicty)
    label_encodery.classes_ = np.append(label_encodery.classes_, '<')
    y_SOS = len(label_encodery.classes_)-1
    label_encodery.classes_ = np.append(label_encodery.classes_, '>')
    y_EOS = len(label_encodery.classes_)-1
    
    
    split = input_string.split()
    
    transformed = list(label_encoderx.transform(split))
    transformed.insert(0, x_SOS)
    transformed.append(x_EOS)
    while len(transformed) < 17:
        transformed.append(x_EOS)
    t = []
    while len(t) < 20:
        t.append(transformed)
          
    transformed = torch.tensor(t)
    
    
   # (20x17)
    layers = 2
    hidden_size = 1000
    dropout = 0.2
    bidirectional = True
    encoder = EncoderRNN(x_EOS+1, hidden_size, layers=layers, 
                     dropout=dropout, bidirectional=bidirectional)
    decoder = DecoderAttn(hidden_size, y_EOS+1, layers=layers, 
                      dropout=dropout, bidirectional=bidirectional)
                      
    # encoder.load_state_dict(torch.load('encoder_80k.pth'))
    # decoder.load_state_dict(torch.load('decoder_80k.pth'))

    encoder.load_state_dict(torch.load(path + 'model-params/' + 'encoder.pth'))
    decoder.load_state_dict(torch.load(path + 'model-params/' + 'decoder.pth'))
            
    #pred = test(transformed, encoder, decoder, y_SOS, label_encodery)
    return test_batch(transformed, encoder, decoder, y_SOS, label_encodery)



def test_batch(input_batch, encoder, decoder, y_SOS, label_encodery):
    
    enc_h_hidden, enc_c_hidden = encoder.create_init_hiddens(input_batch.shape[1])
    enc_hiddens, enc_outputs = encoder(input_batch, enc_h_hidden, enc_c_hidden)
    
    decoder_input = Variable(torch.LongTensor(1,input_batch.shape[1]).fill_(y_SOS))
                             
                             
    dec_h_hidden = enc_outputs[0]
    dec_c_hidden = enc_outputs[1]  

    pred, dec_outputs = decoder(decoder_input, dec_h_hidden, dec_c_hidden, enc_hiddens)
    topv, topi = pred.topk(1,dim=1)
    ni = topi.view(1,-1)
    decoder_input = ni
    dec_h_hidden = dec_outputs[0]
    dec_c_hidden = dec_outputs[1]	
    pred = pred.float()

    return label_encodery.inverse_transform(torch.argmax(pred, dim=1))

            
def construct_dictx():

    with open('dict_xx.csv', newline ='') as f:
        reader = csv.reader(f)
        dictx = list(reader)
        f.close()

    return dictx

def construct_dicty():

    with open('dict_yy.csv', newline ='') as f:
        reader = csv.reader(f)
        dicty = list(reader)
        f.close()
    
    return dicty

  
class NMTDemo():

    def __init__(self, print_results=False):

        self.print_results = print_results
        self.punc = set(string.punctuation)

    def __call__(self, input_pred):

        input_string = input_pred[0]
        str_raw = demo(input_string)
        str_new = ' '.join(char for char in str_raw if char not in self.punc)
        
        if (self.print_results):
            print(f"NMT Translated result: '{str_new}'") 

