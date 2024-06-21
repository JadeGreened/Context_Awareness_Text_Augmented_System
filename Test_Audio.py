import queue
import threading
import time
import wave
import pyaudio
import os

from openai import OpenAI
from pydub import AudioSegment

from viewer import hl2ss, hl2ss_lnm, hl2ss_utilities


class AudioComponent:
    def __init__(self, mainSys, ip):
        self.mainSys = mainSys
        self.ip = ip
        self.profile = hl2ss.AudioProfile.RAW
        self.audio_format = pyaudio.paInt16 if (self.profile == hl2ss.AudioProfile.RAW) else pyaudio.paFloat32
        self.pcmqueue = queue.Queue()
        self.max_wav = 20
        self.file_name_format = r"WavBuffer\wavBuffer_{}"  # 使用原始字符串
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.buffer_cnt = 0
        self.index = 0

    def save_to_local(self):
        while True:
            if self.buffer_cnt < self.max_wav:
                p = pyaudio.PyAudio()
                stream = p.open(format=self.audio_format, channels=hl2ss.Parameters_MICROPHONE.CHANNELS,
                                rate=hl2ss.Parameters_MICROPHONE.SAMPLE_RATE, output=True)
                frames = []
                num_frames = int(48000 / 1536 * 7)  # 7 seconds of audio


                for i in range(num_frames):
                    data = stream.read(self.pcmqueue.get())
                    frames.append(data)

                self.pcmqueue.get()

                filename = self.file_name_format.format(self.index) + ".wav"
                wf = wave.open(filename, 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(48000)
                wf.writeframes(b''.join(frames))
                wf.close()

                stream.stop_stream()
                stream.close()
                p.terminate()

                self.index += 1
                self.buffer_cnt += 1
                print(f"Saved {filename}")
            else:
                # Wait for some files to be processed
                time.sleep(5)

    def audio_to_text(self):
        while True:
            if self.buffer_cnt > 0:
                current_file = self.file_name_format.format(self.index) + ".wav"
                song = AudioSegment.from_wav(current_file)
                song.export("temp_file.wav", format="wav")
                with open("temp_file.wav", "rb") as audio_file:
                    transcription = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
                    print(transcription.text)  # or proper attribute access as needed

                os.remove(current_file)  # Remove the processed file
                os.remove("temp_file.wav")  # Clean up temporary file
                self.buffer_cnt -= 1
                print(f"Processed and removed {current_file}")
            else:
                # Wait for files to be saved
                time.sleep(5)

    def audio_component_on_invoke(self):
        client = hl2ss_lnm.rx_microphone(self.ip, hl2ss.StreamPort.MICROPHONE, profile=self.profile)
        client.open()
        save_thread = threading.Thread(target=self.save_to_local)
        text_thread = threading.Thread(target=self.audio_to_text)
        save_thread.start()
        text_thread.start()

        try:
            while True:
                data = client.get_next_packet()
                audio = hl2ss_utilities.microphone_planar_to_packed(data.payload) if (self.profile != hl2ss.AudioProfile.RAW) else data.payload
                self.pcmqueue.put(audio.tobytes())
        finally:
            client.close()
            save_thread.join()
            text_thread.join()
