import os
from openai import OpenAI
from pydub import AudioSegment



client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

with open("WavBuffer/wav_1.wav", "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )
    # 假设 transcription 是一个对象，需要通过正确的属性访问
    print(transcription.text)  # 或根据实际API调整访问方式
