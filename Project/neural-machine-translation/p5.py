import numpy as np
import string
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch
from torch import optim
from torch.autograd import Variable
import matplotlib.pyplot as plt


data = np.genfromtxt('spa.txt',delimiter='\t', dtype = 'str', encoding = "utf8")
data_x = data[0:80000, 0]
data_y = data[0:80000, 1]
numData = len(data_x)

dictx=[]
for a in range(0, len(data_x)):
    t = data_x[a].split()
    for word in t:
        dictx.append(word)

print('here1')
label_encoderx = LabelEncoder()
label_encoderx.fit(dictx)
label_encoderx.classes_ = np.append(label_encoderx.classes_, '<')

x_SOS = len(label_encoderx.classes_)-1

label_encoderx.classes_ = np.append(label_encoderx.classes_, '>')
x_EOS = len(label_encoderx.classes_)-1

#transformed_x_data = [];



data_x2 = [data.split() for data in data_x]
transformed_x_data = [list(label_encoderx.transform(data)) for data in data_x2]
# for b in range(0, len(data_x)):
#     transformedX = list(label_encoderx.transform(data_x[b].split()))
#     transformed_x_data.append(transformedX)
#     if b%100 ==1:
#         print(b)
# print(transformed_x_data)
# print(' ')

print('here2')
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



data_y2 = [data.split() for data in data_y]
transformed_y_data = [list(label_encodery.transform(data)) for data in data_y2]

# transformed_y_data = [];
# for b in range(0, len(data_y)):
#     transformedY = list(label_encodery.transform(data_y[b].split()))
#     transformed_y_data.append(transformedY)
#     if b%100 ==1:
#         print(b)

print('here3')
x_maxLength = 0;
y_maxLength = 0
for c in range(0, len(transformed_x_data)):
    transformed_x_data[c].insert(0, x_SOS)   
    transformed_x_data[c].append(x_EOS)  
    # u = len(transformed_x_data[c])
    # if u > x_maxLength:
    #     x_maxLength = u
print('here4')    
for c in range(0, len(transformed_y_data)):
    transformed_y_data[c].insert(0, y_SOS)   
    transformed_y_data[c].append(y_EOS)  
    # u = len(transformed_x_data[c])
    # if u > y_maxLength:
    #     y_maxLength = u

maxLength = max(max(map(len, transformed_x_data)), max(map(len, transformed_y_data)))

# train_x = data_x[0:80]
# train_y = data_y[0:80]

# test_x = data_x[81:100]
# test_y = data_y[81:100]



#####



batch_size = 20

num_batches = int(numData/batch_size)
#num_batches_test = int(len(test_x)/batch_size)

batches = list(range(num_batches))
#test_batches = list(range(num_batches_test))


layers = 2
hidden_size = 1000

dropout = 0.2


test_batch_size = 32

epochs = 1

learning_rate= .01

lr_schedule = {}
bidirectional = True;
criterion = nn.NLLLoss()


for batch_number in range(0, num_batches):
    x_tensor = list()
    y_tensor = list()
    xdata = transformed_x_data[batch_number*batch_size:(batch_number+1)*batch_size]
    ydata = transformed_y_data[batch_number*batch_size:(batch_number+1)*batch_size]
    
    # x_maxLength = len(max(xdata, key = len))
    # y_maxLength = len(max(ydata, key = len))
    
    # maxLength = max(x_maxLength, y_maxLength)
    
    x_lens = []    
    y_lens = []       
    for f in range(0, batch_size):
        x_lens.append(len(torch.IntTensor(xdata[f])))
        y_lens.append(len(torch.IntTensor(ydata[f])))
        x_tensor.append(torch.IntTensor(xdata[f]))
        y_tensor.append(torch.IntTensor(ydata[f]))
    padded_X = torch.nn.utils.rnn.pad_sequence(x_tensor,padding_value=x_EOS, batch_first = True)      
    padded_X = torch.nn.utils.rnn.pack_padded_sequence(padded_X,lengths = x_lens,batch_first = True, enforce_sorted = False)
    padded_X = torch.nn.utils.rnn.pad_packed_sequence(sequence=padded_X,batch_first = True, padding_value=x_EOS, total_length= maxLength)
    padded_X = padded_X[0]
    
    
    padded_Y = torch.nn.utils.rnn.pad_sequence(y_tensor,padding_value=y_EOS, batch_first = True)      
    padded_Y = torch.nn.utils.rnn.pack_padded_sequence(padded_Y,lengths = y_lens,batch_first = True, enforce_sorted = False)
    padded_Y = torch.nn.utils.rnn.pad_packed_sequence(sequence=padded_Y,batch_first = True, padding_value=y_EOS, total_length= maxLength)
    padded_Y = padded_Y[0]

    batches[batch_number] = (padded_X, padded_Y)

test_batches = batches[int(len(batches)-.2*len(batches)):len(batches)]
batches = batches[0:int(len(batches)-.2*len(batches)-1)]
# change this later --> taking testbatches that have already been used for training
 

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

# changed
encoder = EncoderRNN(x_EOS+1, hidden_size, layers=layers, 
                     dropout=dropout, bidirectional=bidirectional)
# encoder = EncoderRNN(x_EOS, hidden_size, layers=layers, 
#                      dropout=dropout, bidirectional=bidirectional)
decoder = DecoderAttn(hidden_size, y_EOS+1, layers=layers, 
                      dropout=dropout, bidirectional=bidirectional)

# encoder.load_state_dict(torch.load('encoder.pth'))
# decoder.load_state_dict(torch.load('decoder.pth'))

def train_batch(input_batch, target_batch, encoder, decoder,
                encoder_optimizer, decoder_optimizer, loss_criterion):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0
    enc_h_hidden, enc_c_hidden = encoder.create_init_hiddens(
 	    input_batch.shape[1])

    enc_hiddens, enc_outputs = encoder(input_batch, enc_h_hidden, enc_c_hidden)
    # decoder_input used to have Variable
    
    
    decoder_input = Variable(torch.LongTensor(1, input_batch.shape[1]).fill_(y_SOS))
    #decoder_input = 
    

    dec_h_hidden = enc_outputs[0]
    dec_c_hidden = enc_outputs[1]
    for i in range(target_batch.shape[0]):
        
    
        
        pred, dec_outputs = decoder(decoder_input, dec_h_hidden, dec_c_hidden, enc_hiddens)
        decoder_input = target_batch[i].view(1,-1)
        dec_h_hidden = dec_outputs[0]
        dec_c_hidden = dec_outputs[1]
        
        #=output is a bigger sentence than input
        # and it doesnt like it
        #pred = torch.argmax(pred, dim = 1)
        #pred = pred.type(torch.LongTensor)
        
        
        #tb = target_batch[i].type(torch.LongTensor)
        p = pred.float()
        tb = target_batch[i].type(torch.LongTensor)
        loss += loss_criterion(p,tb)
        #loss += loss_criterion(pred,tb)


    loss.backward()

    # torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.clip)
    # torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_batch.shape[0]


def train(train_batches, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_criterion):
    round_loss = 0
    i = 1
    for batch in train_batches:
        i += 1
        input_batch = batch[0]
        target_batch = batch[1]
        batch_loss = train_batch(input_batch, target_batch, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_criterion)
        round_loss += batch_loss
        print(i)
    return (round_loss/len(train_batches))
      
t_loss = [] 
iteration= []
for i in range(epochs):
    encoder.train()
    decoder.train()
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    train_loss = train(batches, encoder, decoder, encoder_optimizer, 
                        decoder_optimizer, criterion)
    t_loss.append(train_loss)
    iteration.append(i+1)
    print(train_loss)

plt.figure()
plt.plot(iteration, t_loss)
plt.title('Loss as a Function of Epoch')
plt.xlabel('Epoch Number')
plt.ylabel('Training Loss')

torch.save(decoder.state_dict(), 'decoder_80k_5ep.pth')
torch.save(encoder.state_dict(), 'encoder_80k_5ep.pth')
#TESTING
def test_batch(input_batch, target_batch, encoder, decoder, loss_criterion):
    loss = 0
    
    enc_h_hidden, enc_c_hidden = encoder.create_init_hiddens(input_batch.shape[1])
    enc_hiddens, enc_outputs = encoder(input_batch, enc_h_hidden, enc_c_hidden)
    
    decoder_input = Variable(torch.LongTensor(1,input_batch.shape[1]).fill_(y_SOS))
                             
                             
    dec_h_hidden = enc_outputs[0]
    dec_c_hidden = enc_outputs[1]  
    
    for i in range(target_batch.shape[0]):
        pred, dec_outputs = decoder(decoder_input, dec_h_hidden, dec_c_hidden, enc_hiddens)
        topv, topi = pred.topk(1,dim=1)
        ni = topi.view(1,-1)
        decoder_input = ni
        dec_h_hidden = dec_outputs[0]
        dec_c_hidden = dec_outputs[1]	
        pred = pred.float()
        tb = target_batch[i].type(torch.LongTensor)
        loss += loss_criterion(pred,tb)
        if i == target_batch.shape[0]-1:
            print(' ')
            print('Predicted')
            print(label_encodery.inverse_transform(torch.argmax(pred, dim=1)))
            print('Actual')
            print(label_encodery.inverse_transform(tb))
            
        
        
        
    return loss.item()/target_batch.shape[0]

def test(test_batches, encoder, decoder, loss_criterion):
    test_loss= 0
    i = 1
    for batch in test_batches:
        input_batch = batch[0]
        target_batch = batch[1]
        batch_loss = test_batch(input_batch, target_batch, encoder, decoder, loss_criterion)
        test_loss += batch_loss
        print(i)
        i += 1
    print(test_loss/len(test_batches)) 
    return (test_loss/len(test_batches))

test_loss = test(test_batches, encoder, decoder, criterion)











 
