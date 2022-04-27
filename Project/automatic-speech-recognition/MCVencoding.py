import torch
import string
import unidecode
from typing import List

class EncoderDecoder:

    def __init__(self) -> None:
        # '-' represents space, '_' represents blank
        self.char_map = {'-':0,'a':1,'b':2,'c':3,'d':4,'e':5,'f':6,'g':7,'h':8,'i':9,'j':10,'k':11,'l':12,'m':13,'n':14,'o':15,'p':16,'q':17,'r':18,'s':19,'t':20,'u':21,'v':22,'w':23,'x':24,'y':25,'z':26,'_':27}
        self.int_map = {}
        for char in self.char_map:
            self.int_map[self.char_map[char]] = char

    def integer_encoding(self, sentence_str: str) -> List[int]:

        # remove punctuation, remove accents, make all lowercase, replace spaces with dashes, and convert to character array
        sentence_str = ''.join(char for char in sentence_str if char not in set(string.punctuation+"\u2019"+"\u2018"+"\u2013"+"\u201C"+"\u201D"))
        sentence_str = sentence_str.strip(chr(34))
        sentence_str = unidecode.unidecode(sentence_str)
        sentence_str = sentence_str.lower()
        sentence_str = sentence_str.replace(' ','-')
        sentence_list = list(sentence_str)

        integer_list = [self.char_map[char] for char in sentence_list]

        return integer_list

    def label_decoding(self, labels: torch.Tensor, blank_label: int=27) -> List[str]:

        label_strs = list()

        for char_ints in labels.int().tolist():
            label_strs.append(self.integer_decoding(char_ints, blank_label=blank_label))

        return label_strs

    def greedy_output_decoding(self, outputs: torch.Tensor, blank_int: int=27, remove_repetitions: bool=True) -> List[str]:

        
        output_strs = list()
        # print(outputs.shape)        
        # print(torch.argmax(outputs,dim=2).shape)
        # print(torch.argmax(outputs,dim=2).squeeze(1).shape)
        char_arg_max = torch.argmax(outputs,dim=2).squeeze(1)
        for char_out in char_arg_max:
            prev_int = -1
            char_ints = list()
            for int in char_out:
                if (int.item() != blank_int):
                    if (remove_repetitions and int.item() == prev_int):
                        continue
                    char_ints.append(int.item())
                prev_int = int.item()
            output_strs.append(self.integer_decoding(char_ints))
        return output_strs

    def integer_decoding(self, integer_list: List[int], blank_label: int=27) -> str:

        sentence_list = [self.int_map[int] for int in integer_list if int != blank_label]
        sentence_str = ''.join(char for char in sentence_list)
        sentence_str = sentence_str.replace('-',' ')
        sentence_str = ' '.join(sentence_str.split())

        return sentence_str
