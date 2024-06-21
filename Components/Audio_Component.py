
import soundcard as sc
import queue
import threading
import time
import wave
import pyaudio
import os
import lameenc
from openai import OpenAI
from pydub import AudioSegment
import numpy as np

from viewer import hl2ss
from viewer import hl2ss_utilities
from viewer import hl2ss_lnm


class AudioComponent():
    def __init__(self, mainSys, ip):
        self.mainSys = mainSys
        self.ip = ip
        self.profile = hl2ss.AudioProfile.RAW
        self.audio_format = pyaudio.paInt16 if (self.profile == hl2ss.AudioProfile.RAW) else pyaudio.paFloat32
        self.pcmqueue1 = queue.Queue()
        self.max_wav = 20
        self.file_name_format = r"WavBuffer\Wav_{}"
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.buffer_cnt = 0
        self.index = 0
        self.channel = hl2ss.Parameters_MICROPHONE.ARRAY_TOP_LEFT
        self.read_index = 0
        self.file_duration = 10


        #Audio Recorder
        self.frames = []
        self.pcmqueue2 = queue.Queue()
        self.encoder = lameenc.Encoder()
        self.encoder.set_bit_rate(128)
        self.encoder.set_in_sample_rate(48000)
        self.encoder.set_channels(2)
        self.encoder.set_quality(2)
        self.lock = threading.Lock()
        self.file_index = 0

    def play_and_save_audio(self):
        p = pyaudio.PyAudio()
        format = pyaudio.paInt16
        channels = 2
        rate = 48000

        stream = p.open(format=format, channels=channels, rate=rate, output=True)
        stream.start_stream()

        start_time = time.time()
        wf = self.create_new_wav_file()

        try:
            while True:
                data = self.pcmqueue1.get()
                stream.write(data)
                wf.writeframes(data)

                if time.time() - start_time >= self.file_duration:
                    wf.close()
                    wf = self.create_new_wav_file()
                    start_time = time.time()

        finally:
            stream.stop_stream()
            stream.close()
            wf.close()
            p.terminate()


    def create_new_wav_file(self):
        filename = fr"WavBuffer\wav_{self.file_index}.wav"
        self.file_index += 1  # 递增文件编号

        wf = wave.open(filename, 'wb')
        wf.setnchannels(2)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(48000)
        print(f"New file created: {filename}")
        return wf



    def audio_to_text(self):
        print("The process has been started")
        while True:
            current_file = self.file_name_format.format(self.read_index) + ".wav"
            try:
                with open(current_file, "rb") as audio_file:
                    transcription = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
                    print(transcription.text)
                    text_to_write = "The user is speaking:" + transcription.text
                    self.mainSys.read_then_write(text_to_write)
                print(f"Processed and removed {current_file}")
                os.remove(current_file)
                self.read_index += 1
            except FileNotFoundError:
                print(f"File not found: {current_file}, will retry in 10 seconds...")
                time.sleep(10)
            except Exception as e:
                print("Something went wrong , may be connetion error")
                time.sleep(1)

    def audio_component_on_invoke(self):
        client = hl2ss_lnm.rx_microphone(self.ip, hl2ss.StreamPort.MICROPHONE, profile=self.profile)
        client.open()
        play_thread = threading.Thread(target=self.play_and_save_audio)
        # record_thread = threading.Thread(target=self.record_audio)
        # save_thread = threading.Thread(target=self.save_to_local)

        text_thread = threading.Thread(target=self.audio_to_text)
        play_thread.start()
        # record_thread.start()
        # save_thread.start()
        text_thread.start()

        while True:
            data = client.get_next_packet()
            audio = hl2ss_utilities.microphone_planar_to_packed(data.payload) if (self.profile != hl2ss.AudioProfile.RAW) else data.payload
            self.pcmqueue1.put(audio.tobytes())

