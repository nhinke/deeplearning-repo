import sys
import time
import wave
import pyaudio
import threading

import torch
import torchaudio

import numpy as np
from threading import Event

from model3 import ASR
from MCVdataset import DataPreprocesser

# adapted from https://github.com/LearnedVector/A-Hackers-AI-Voice-Assistant/blob/master/VoiceAssistant/speechrecognition/engine.py

class Listener:

    def __init__(self, sample_rate=48000, record_seconds=4):
        self.chunk = 1024
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.p = pyaudio.PyAudio()
        self.stream = self.open_stream()

    def listen(self, queue):
        while True:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            queue.append(data)
            time.sleep(0.01)

    def listen_once(self):
        self.close()
        self.stream = self.open_stream()
        aud_queue = list()
        start_time = time.time()
        while (time.time() - start_time < self.record_seconds):
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            aud_queue.append(data)
            # time.sleep(0.001)
        return aud_queue

    def open_stream(self):
        return self.p.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, input=True, output=True, frames_per_buffer=self.chunk, input_device_index=8)

    def close(self):
        self.stream.close()

    def run(self, queue):
        thread = threading.Thread(target=self.listen, args=(queue,), daemon=True)
        thread.start()
        print("\Speech Recognition engine is now listening... \n")


class SpeechRecognitionEngine:

    def __init__(self, sample_rate=48000, context_length=10):
        self.sample_rate = sample_rate
        self.listener = Listener(sample_rate=self.sample_rate)
        self.model = ASR()
        self.model.load_saved_model()
        self.model.eval().to(torch.device('cpu'))  #run on cpu
        self.data_preprocesser = DataPreprocesser(augment=False, resample_audio=True)
        
        self.audio_q = list()
        # self.beam_results = ""
        self.out_args = None
        self.context_length = context_length * 50 # multiply by 50 because each 50 from output frame is 1 second
        self.start = False

    def save(self, waveforms, fname="audio_temp"):
        wf = wave.open(fname, "wb")
        # set the channels
        wf.setnchannels(1)
        # set the sample format
        wf.setsampwidth(self.listener.p.get_sample_size(pyaudio.paInt16))
        # set the sample rate
        wf.setframerate(self.sample_rate)
        # write the frames as bytes
        wf.writeframes(b"".join(waveforms))
        # close the file
        wf.close()
        return fname

    def predict(self, audio):
        # with torch.no_grad():
        fname = self.save(audio)
        waveform, sample_rate = torchaudio.load(fname)  # don't normalize on train
        input = self.data_preprocesser.get_mel_spectrogram(waveform, sample_rate)
        # input_length = input.size(2) # seq_length
        # print(input.shape)
        input_length = torch.IntTensor(1).fill_(input.size(2))
        output_probs = self.model.inference_step(input.unsqueeze(1), input_length)
        pred_output_str = self.model.greedy_decoding(output_probs)
        beam_output_str = self.model.beam_decoding(output_probs)

        beam_results = beam_output_str
        greedy_results = pred_output_str

        # out, self.hidden = self.model(log_mel, self.hidden)
        # out = torch.nn.functional.softmax(out, dim=2)
        # out = out.transpose(0, 1)
        # self.out_args = out if self.out_args is None else torch.cat((self.out_args, out), dim=1)
        # results = self.beam_search(self.out_args)

        # current_context_length = results.shape[1] / 50  # in seconds

        # current_context_length = self.out_args.shape[1] / 50  # in seconds
        # if self.out_args.shape[1] > self.context_length:
        #     self.out_args = None
        # return results, current_context_length
        return beam_results, greedy_results

    def inference_loop(self, action):
        while True:
            if len(self.audio_q) < 5:
                continue
            else:
                pred_q = self.audio_q.copy()
                self.audio_q.clear()
                action(self.predict(pred_q))
            time.sleep(0.05)

    def run(self, action):
        self.listener.run(self.audio_q)
        thread = threading.Thread(target=self.inference_loop, args=(action,), daemon=True)
        thread.start()


class DemoAction:

    def __init__(self, print_aggregate_transcript=True, print_greedy_results=False):
        self.asr_results = ''
        # self.current_beam = ""
        self.print_transcript = print_aggregate_transcript
        self.print_greedy_results = print_greedy_results

    def __call__(self, x):
        beam_results, greedy_results = x
        beam_results_str = ''.join(beam_results)
        greedy_results_str = ''.join(greedy_results)
        # results, current_context_length = x
        # self.current_beam = results
        # print(results)
        # print(type(results))
        # print(len(results))
        # transcript = self.asr_results + results
        transcript = ' '.join(self.asr_results.split() + beam_results_str.split())
        self.asr_results = transcript
        
        # print()

        if (self.print_greedy_results):
            print(f"Greedy results:  '{greedy_results_str}'")

        print(f"CTCBeam results: '{beam_results_str}'")

        if (self.print_transcript):
            print(f"Total transcript: '{transcript}'")

        # if current_context_length > 10:
        #     self.asr_results = transcript

def main():

    # asr_engine = SpeechRecognitionEngine()
    # action = DemoAction()
    # asr_engine.run(action)
    # threading.Event().wait()

    recording_time = 4
    audio_sample_rate = 48000

    asr_engine = SpeechRecognitionEngine(sample_rate=audio_sample_rate)
    asr_demo = DemoAction(print_aggregate_transcript=False, print_greedy_results=True)
    py_listener = Listener(sample_rate=audio_sample_rate, record_seconds=recording_time)

    while True:

        inp = input(f"\nPress enter to start recording for {recording_time} seconds or type 'q' to quit...")
        if (inp == 'q' or inp == ' q'):
            py_listener.close()
            return

        # audio_queue = list().copy()
        audio_queue = py_listener.listen_once()
        print('Finished recording! Now processing audio...')
        prediction = asr_engine.predict(audio_queue.copy())
        asr_demo(prediction)


if __name__ == "__main__":
    main()

