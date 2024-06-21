import queue

import pyaudio
import requests
import os

from openai import OpenAI
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import clip
import google.generativeai as genai
from pynput import keyboard

import threading

from Components.Eyetracking_Component import Eyetracking_Component
from Components.Video_Component import VideoComponent
from Components.IMU_Component import IMU_Component
from Components.Audio_Component import AudioComponent
from viewer import hl2ss
from viewer import hl2ss_lnm
from viewer import hl2ss_rus
import threading
import socket
import time
from pymilvus import connections, db, MilvusClient, Milvus, model
from urllib.parse import unquote_plus, parse_qs


# To get the video stream
class VideoGraper:
    def __init__(self, ip):
        self._ip = ip

    def get_video_client(self):
        # Settings --------------------------------------------------------------------
        # HoloLens address
        host = self._ip
        # Operating mode
        # 0: video
        # 1: video + camera pose
        # 2: query calibration (single transfer)
        mode = hl2ss.StreamMode.MODE_0
        # Enable Mixed Reality Capture (Holograms)
        enable_mrc = True
        # Camera parameters
        width = 1920
        height = 1080
        framerate = 30
        # Framerate denominator (must be > 0)
        # Effective FPS is framerate / divisor
        divisor = 1

        # Video encoding profile
        profile = hl2ss.VideoProfile.H265_MAIN

        # Decoded format
        # Options include:
        # 'bgr24'
        # 'rgb24'
        # 'bgra'
        # 'rgba'
        # 'gray8'
        decoded_format = 'bgr24'
        # ------------------------------------------------------------------------------
        hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, enable_mrc=enable_mrc)
        client = hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, mode=mode, width=width, height=height,
                                 framerate=framerate, divisor=divisor, profile=profile, decoded_format=decoded_format)
        return client


class AudioGraper:
    def __init__(self, ip):
        self._ip = ip

    def pcmworker(self, pcmqueue):
        enable = True
        p = pyaudio.PyAudio()
        stream = p.open(format=self.audio_format, channels=1, rate=hl2ss.Parameters_MICROPHONE.SAMPLE_RATE, output=True)
        stream.start_stream()
        while (enable):
            stream.write(pcmqueue.get())
        stream.stop_stream()
        stream.close()

    def get_audio_client(self):
        # Channel
        # Options:
        # hl2ss.Parameters_MICROPHONE.ARRAY_TOP_LEFT
        # hl2ss.Parameters_MICROPHONE.ARRAY_TOP_CENTER
        # hl2ss.Parameters_MICROPHONE.ARRAY_TOP_RIGHT
        # hl2ss.Parameters_MICROPHONE.ARRAY_BOTTOM_LEFT
        # hl2ss.Parameters_MICROPHONE.ARRAY_BOTTOM_RIGHT
        self.channel = hl2ss.Parameters_MICROPHONE.ARRAY_TOP_LEFT

        # ------------------------------------------------------------------------------

        self.audio_format = pyaudio.paFloat32
        self.pcmqueue = queue.Queue()
        thread = threading.Thread(target=self.pcmworker, args=(self.pcmqueue,))
        thread.start()
        client = hl2ss_lnm.rx_microphone(self._ip, hl2ss.StreamPort.MICROPHONE, profile=hl2ss.AudioProfile.RAW,
                                         level=hl2ss.AACLevel.L5)
        return client


class IMUGraper:
    def __init__(self, ip):
        self._ip = ip


    def get_imu_client(self):
        # Port
        # Options:
        # hl2ss.StreamPort.RM_IMU_ACCELEROMETER
        # hl2ss.StreamPort.RM_IMU_GYROSCOPE
        # hl2ss.StreamPort.RM_IMU_MAGNETOMETER
        port = hl2ss.StreamPort.RM_IMU_ACCELEROMETER

        # Operating mode
        # 0: samples
        # 1: samples + rig pose
        # 2: query calibration (single transfer)
        mode = hl2ss.StreamMode.MODE_1
        client = hl2ss_lnm.rx_rm_imu(self._ip, port, mode=mode)
        return client

class EyegazeGraper:
    def __init__(self, ip):
        self._ip = ip

    def get_eyegaze_client(self):
        # Target Frame Rate
        # Options: 30, 60, 90
        fps = 30
        client = hl2ss_lnm.rx_eet(self._ip, hl2ss.StreamPort.EXTENDED_EYE_TRACKER, fps=fps)
        return client


# The AI Agent for data processing
class Agent:
    def __init__(self):
        # 配置 gpt3.5 并获取 API 键
        self.llm = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        # Load the model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_version = "ViT-B/32"
        self.clip_feat_dim = \
        {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768, 'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512,
         'ViT-L/14': 768}[clip_version]
        self.model, self.preprocess = clip.load(clip_version)  # clip.available_models()

    def get_text_feats(self, in_text, batch_size=64):
        text_tokens = clip.tokenize(in_text).to(self.device)
        text_id = 0
        text_feats = np.zeros((len(in_text), self.clip_feat_dim), dtype=np.float32)
        while text_id < len(text_tokens):  # Batched inference.
            batch_size = min(len(in_text) - text_id, batch_size)
            text_batch = text_tokens[text_id:text_id + batch_size]
            with torch.no_grad():
                batch_feats = self.model.encode_text(text_batch).float()
            batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
            batch_feats = np.float32(batch_feats.cpu())
            text_feats[text_id:text_id + batch_size, :] = batch_feats
            text_id += batch_size
        return text_feats

    def get_img_feats(self, img):
        img_pil = Image.fromarray(np.uint8(img))
        img_in = self.preprocess(img_pil)[None, ...]
        with torch.no_grad():
            img_feats = self.model.encode_image(img_in.cuda()).float()
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        img_feats = np.float32(img_feats.cpu())
        return img_feats

    def get_nn_text(self, raw_texts, text_feats, img_feats):
        scores = text_feats @ img_feats.T
        scores = scores.squeeze()
        high_to_low_ids = np.argsort(scores).squeeze()[::-1]
        high_to_low_texts = [raw_texts[i] for i in high_to_low_ids]
        high_to_low_scores = np.sort(scores).squeeze()[::-1]
        return high_to_low_texts, high_to_low_scores

    def download_file(self, url, filename):
        if not os.path.exists(filename):
            response = requests.get(url)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)
            else:
                print(f"Failed to download the file: {url}")

    def prompt_llm(self, prompt):
        # 调用模型生成完成
        completion = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        # 返回生成的内容
        return completion.choices[0].message.content

    def load_files(self):
        # Load scene categories from Places365.
        # if not os.path.exists('categories_places365.txt'):
        #   download_file("https://raw.githubusercontent.com/zhoubolei/places_devkit/master/categories_places365.txt","categories_places365.txt")
        place_categories = np.loadtxt('categories_places365.txt', dtype=str)
        self.place_texts = []
        for place in place_categories[:, 0]:
            place = place.split('/')[2:]
            if len(place) > 1:
                place = place[1] + ' ' + place[0]
            else:
                place = place[0]
            place = place.replace('_', ' ')
            self.place_texts.append(place)
        self.place_feats = self.get_text_feats([f'Photo of a {p}.' for p in self.place_texts])

        # Load object categories from Tencent ML Images.
        # if not os.path.exists('dictionary_and_semantic_hierarchy.txt'):
        #   download_file("https://raw.githubusercontent.com/Tencent/tencent-ml-images/master/data/dictionary_and_semantic_hierarchy.txt","dictionary_and_semantic_hierarchy.txt")
        with open('dictionary_and_semantic_hierarchy.txt') as fid:
            object_categories = fid.readlines()
        self.object_texts = []
        for object_text in object_categories[1:]:
            object_text = object_text.strip()
            object_text = object_text.split('\t')[3]
            safe_list = ''
            for variant in object_text.split(','):
                text = variant.strip()
                safe_list += f'{text}, '
            safe_list = safe_list[:-2]
            if len(safe_list) > 0:
                self.object_texts.append(safe_list)
        self.object_texts = [o for o in list(set(self.object_texts)) if
                             o not in self.place_texts]  # Remove redundant categories.
        self.object_feats = self.get_text_feats([f'Photo of a {o}.' for o in self.object_texts])
        self.BASE_URL = 'http://212.64.10.189'
        # you can fill in the blank with you own service ip, here is default
        self.milvus_client = MilvusClient()
        self.embedding_model = model.DefaultEmbeddingFunction()
        self.milvus_client.load_collection("thoughts")

    def process_data(self, img):
        verbose = True

        img_feats = self.get_img_feats(img)
        plt.imshow(img)
        plt.show()

        # Zero-shot VLM: classify image type.
        img_types = ['photo', 'cartoon', 'sketch', 'painting']
        img_types_feats = self.get_text_feats([f'This is a {t}.' for t in img_types])
        sorted_img_types, img_type_scores = self.get_nn_text(img_types, img_types_feats, img_feats)
        img_type = sorted_img_types[0]

        # Zero-shot VLM: classify number of people.
        ppl_texts = ['no people', 'people']
        ppl_feats = self.get_text_feats([f'There are {p} in this photo.' for p in ppl_texts])
        sorted_ppl_texts, ppl_scores = self.get_nn_text(ppl_texts, ppl_feats, img_feats)
        ppl_result = sorted_ppl_texts[0]
        if ppl_result == 'people':
            ppl_texts = ['is one person', 'are two people', 'are three people', 'are several people', 'are many people']
            ppl_feats = self.get_text_feats([f'There {p} in this photo.' for p in ppl_texts])
            sorted_ppl_texts, ppl_scores = self.get_nn_text(ppl_texts, ppl_feats, img_feats)
            ppl_result = sorted_ppl_texts[0]
        else:
            ppl_result = f'are {ppl_result}'

        # Zero-shot VLM: classify places.
        place_topk = 3
        sorted_places, places_scores = self.get_nn_text(self.place_texts, self.place_feats, img_feats)

        # Zero-shot VLM: classify objects.
        obj_topk = 10
        sorted_obj_texts, obj_scores = self.get_nn_text(self.object_texts, self.object_feats, img_feats)
        object_list = ', '.join(sorted_obj_texts[:obj_topk])

        # Zero-shot LM: generate captions.
        num_captions = 10
        prompt = f"""I am an intelligent image captioning bot.
        This image is a {img_type}. There {ppl_result}.
        I think this photo was taken at a {sorted_places[0]}, {sorted_places[1]}, or {sorted_places[2]}.
        I think there might be a {object_list} in this {img_type}.
        A creative short caption I can generate to describe this image is:"""
        caption_texts = [self.prompt_llm(prompt) for _ in range(num_captions)]

        # Zero-shot VLM: rank captions.
        caption_feats = self.get_text_feats(caption_texts)
        sorted_captions, caption_scores = self.get_nn_text(caption_texts, caption_feats, img_feats)
        print(f'{sorted_captions[0]}\n')

        if verbose:
            print('VLM: This image is a:')
            for img_type, score in zip(sorted_img_types, img_type_scores):
                print(f'{score:.4f} {img_type}')

            print('\nVLM: There:')
            for ppl_text, score in zip(sorted_ppl_texts, ppl_scores):
                print(f'{score:.4f} {ppl_text}')

            print('\nVLM: I think this photo was taken at a:')
            for place, score in zip(sorted_places[:place_topk], places_scores[:place_topk]):
                print(f'{score:.4f} {place}')

            print('\nVLM: I think there might be a:')
            for obj_text, score in zip(sorted_obj_texts[:obj_topk], obj_scores[:obj_topk]):
                print(f'{score:.4f} {obj_text}')

            print(f'\nLM generated captions ranked by VLM scores:')
            for caption, score in zip(sorted_captions, caption_scores):
                print(f'{score:.4f} {caption}')

        return sorted_captions[0]

    def save_to_database(self, collection_name, thoughts):
        vector = self.embedding_model.encode_documents(thoughts)
        print("Vector dimensions:", len(vector[0]))
        thought = thoughts[0]
        data = [
            {"vector": vector[0], "thought": thought}
        ]
        print("Data to insert:", data)
        self.milvus_client.insert(collection_name=collection_name, data=data)


    def send_data(self, message):
        """发送数据到队列，并检查请求是否成功"""
        url = f"{self.BASE_URL}/python"
        try:
            response = requests.post(url, data=message)
            # 检查HTTP状态码是否表示成功（200-299）
            if response.ok:
                print("Sent! The prediction is: ", message)
            else:
                # 你可以选择记录日志、抛出异常或返回错误消息
                return f"Failed to send data: {response.status_code}, {response.text}"
        except requests.RequestException as e:
            # 网络问题或请求被拒绝等
            return f"Error sending data: {str(e)}"

    def get_data(self):
        """从队列获取数据，并检查请求是否成功"""
        url = f"{self.BASE_URL}/unity"
        response = requests.get(url)

        if response.status_code == 200:
            # 假设数据以 application/x-www-form-urlencoded 形式返回
            # 首先解码URL编码
            decoded_response = unquote_plus(response.text)

            # 如果确实需要解析键值对
            parsed_data = parse_qs(decoded_response)

            # 假设你期望得到 'message' 键的数据
            if 'message' in parsed_data:
                message = parsed_data['message'][0]  # 获取列表中的第一个元素
                print(f"Received message: {message}")
                return message
            else:
                print("No message key found in the response.")
        else:
            print(f"Failed to fetch data: {response.status_code}")

    def clear_queue(self, queue_name):
        """ 清空指定队列 """
        url = f"{self.BASE_URL}/clear/{queue_name}"
        response = requests.post(url)
        print(f"Clearing {queue_name} - Status Code: {response.status_code}")




def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable


def agent_on_invoke(agent, img):
    global thread_created
    thought = agent.process_data(img)
    thoughts = [thought]
    agent.save_to_database("thoughts", thoughts)
    time.sleep(2)
    with lock:
        thread_created = False

# search in the vectore database for the prediction
def make_predictions(agent):
    while True:
        typed_data = agent.get_data()
        if not typed_data:
            print("Received empty data, skipping...")
            time.sleep(5)
            continue

        # Ensure the collection exists
        if not agent.milvus_client.has_collection("thoughts"):
            print("Collection 'thoughts' does not exist.")
            continue

        # Encode the typed data into query vectors
        datalist = [typed_data]
        query_vector = agent.embedding_model.encode_queries(datalist)

        # Execute the search
        prediction = agent.milvus_client.search(
            collection_name="thoughts",  # target collection
            data=query_vector,  # query vectors
            limit=2,  # number of returned entities
            output_fields=["thought"],  # specifies fields to be returned
        )

        # Handle the search results
        if prediction:
            # 首先检查是否有结果
            if len(prediction) > 0 and len(prediction[0]) > 0:
                # 然后检查第一个结果中是否有数据
                if "entity" in prediction[0][0] and "thought" in prediction[0][0]["entity"]:
                    predicted_sentence = prediction[0][0]["entity"]["thought"]
                    print(predicted_sentence)
                    if predicted_sentence:  # 确保 predicted_sentence 不为空
                        agent.send_data(predicted_sentence)
                    else:
                        print("Predicted sentence is empty, skipping database storage.")
                else:
                    print("Entity or thought field missing in results.")
            else:
                print("No results returned, skipping this iteration.")
        else:
            print("Invalid or incomplete prediction received, skipping this iteration.")

        time.sleep(2)










lock = threading.Lock()
thread_created = False


# if __name__ == '__main__':
#     ip = "192.168.3.34"
#     thread_created = False
#     enable = True
#     enable_Video = True
#     enable_Eyetracking = True
#     enable_IMU = True
#     enable_Audio = True
#
#     # agent = Agent()
#     # agent.load_files()
#     # agent.clear_queue("python")
#     # agent.clear_queue("unity")
#
#     # 视频
#     if enable_Video:
#         vg = VideoGraper(ip)
#         video_client = vg.get_video_client()
#         video_client.open()
#
#     # 眼动追踪
#     if enable_Eyetracking:
#         eg = EyegazeGraper(ip)
#         eyegaze_client = eg.get_eyegaze_client()
#         eyegaze_client.open()
#
#     # 惯性测量单元（IMU）
#     if enable_IMU:
#         imu = IMUGraper(ip)
#         imu_client = imu.get_imu_client()
#         imu_client.open()
#
#     # 音频
#     if enable_Audio:
#         ag = AudioGraper(ip)
#         audio_client = ag.get_audio_client()
#         audio_client.open()
#
#     listener = keyboard.Listener(on_press=on_press)
#     listener.start()
#
#     # #get the audio array
#     # audio_thread = threading.Thread(target=ag.pcmworker, args=(ag.pcmqueue,))
#     # audio_thread.start()
#
#     # start the prediction task
#     # background_thread = threading.Thread(target=make_predictions,args=(agent,))
#     # background_thread.start()
#
#
#
#
#     while (enable):
#         if enable_Video:
#             try:
#                 video_data = video_client.get_next_packet()
#                 cv2.imshow('Video', video_data.payload.image)
#                 cv2.waitKey(1)
#             except Exception as e:
#                 print(f"Error fetching video packet: {e}")
#                 enable = False  # Optionally stop the loop if critical error occurs
#
#         if enable_Eyetracking:
#             try:
#                 eyegaze_data = eyegaze_client.get_next_packet()
#                 eet = hl2ss.unpack_eet(eyegaze_data.payload)
#                 print("-------------------------------------Eye tracking-------------------------------------")
#                 print(f'Tracking status at time {eyegaze_data.timestamp}')
#                 print('Pose')
#                 print(eyegaze_data.pose)
#                 print(f'Calibration: Valid={eet.calibration_valid}')
#                 print(
#                     f'Combined eye gaze: Valid={eet.combined_ray_valid} Origin={eet.combined_ray.origin} Direction={eet.combined_ray.direction}')
#                 print(
#                     f'Left eye gaze: Valid={eet.left_ray_valid} Origin={eet.left_ray.origin} Direction={eet.left_ray.direction}')
#                 print(
#                     f'Right eye gaze: Valid={eet.right_ray_valid} Origin={eet.right_ray.origin} Direction={eet.right_ray.direction}')
#                 print(f'Left eye openness: Valid={eet.left_openness_valid} Value={eet.left_openness}')
#                 print(f'Right eye openness: Valid={eet.right_openness_valid} Value={eet.right_openness}')
#                 print(f'Vergence distance: Valid={eet.vergence_distance_valid} Value={eet.vergence_distance}')
#             except Exception as e:
#                 print(f"Error fetching eyetracking data: {e}")
#
#         if enable_IMU:
#             try:
#                 imu_data = imu_client.get_next_packet()
#                 print("------------------------------------------IMU-----------------------------------------")
#                 print(f'Pose at time {imu_data.timestamp}')
#                 print(imu_data.pose)
#
#                 imu = hl2ss.unpack_rm_imu(imu_data.payload)
#                 count = imu.get_count()
#                 sample = imu.get_frame(0)
#                 print(f'Got {count} samples at time {imu_data.timestamp}')
#                 print(
#                     f'First sample: sensor_ticks={sample.vinyl_hup_ticks} soc_ticks={sample.soc_ticks} x={sample.x} y={sample.y} z={sample.z} temperature={sample.temperature}')
#             except Exception as e:
#                 print(f"Error fetching IMU data: {e}")
#
#         # if enable_Audio:
#         #     try:
#         #         audio_data = audio_client.get_next_packet()
#         #         audio = audio_data.payload[0, ag.channel::hl2ss.Parameters_MICROPHONE.ARRAY_CHANNELS]
#         #         ag.pcmqueue.put(audio.tobytes())
#         #         print("------------------------------------------audio-----------------------------------------")
#         #     except Exception as e:
#         #         print(f"Error fetching audio data: {e}")
#
#         # if not thread_created:
#         #     with lock:
#         #         if not thread_created:
#         #             t = threading.Thread(target=agent_on_invoke, args=(agent, video_data.payload.image))
#         #             t.start()
#         #             thread_created = True
#
#     video_client.close()
#     eyegaze_client.close()
#     imu_client.close()
#     ag.pcmqueue.put(b'')
#     # audio_thread.join()
#     listener.join()
#
#     hl2ss_lnm.stop_subsystem_pv(vg.get_ip(), hl2ss.StreamPort.PERSONAL_VIDEO)


class MainSys:
    def __init__(self):
        self.file_name = "database.txt"
        self.lock = threading.Lock()

    def read_file(self):
        with self.lock:
            try:
                with open(self.file_name, 'r') as file:
                    data = file.read()
                return data
            except FileNotFoundError:
                print("Error: The file does not exist.")
                return None
            except IOError as e:
                print(f"IO Error occurred: {e}")
                return None

    def write_file(self, text):
        with self.lock:
            try:
                with open(self.file_name, 'a') as file:
                    file.write(text)
            except IOError as e:
                print(f"IO Error occurred: {e}")

    def read_then_write(self, text):
        # 先读取文件内容
        data = self.read_file()
        if data is not None:
            print("Read from file:", data)
            # 读取的数据展示后, 只需写入新数据
            self.write_file(text + "\n")  # 确保文本换行
            print("Written to file:", text)
        else:
            # 如果文件不存在或读取失败，直接写入新数据
            self.write_file(text + "\n")
            print("Written to file:", text)


if __name__ == "__main__":
    mainSys = MainSys()
    # change your ip here
    ip = "192.168.3.34"
    process_thread_circle = 1
    video_component = VideoComponent(ip, mainSys)
    imu_Component = IMU_Component(ip, mainSys)

    video_thread = threading.Thread(target=video_component.video_component_on_invoke)
    video_thread.start()

    imu_thread = threading.Thread(target=imu_Component.imu_component_on_invoke)
    imu_thread.start()
    audio_component = AudioComponent(mainSys,ip)

    audio_thread = threading.Thread(target=audio_component.audio_component_on_invoke)
    audio_thread.start()
    eyetracking_component = Eyetracking_Component(mainSys,ip)
    eyetracking_thread = threading.Thread(target=eyetracking_component.eye_tracking_component_on_invoke)
    eyetracking_thread.start()


