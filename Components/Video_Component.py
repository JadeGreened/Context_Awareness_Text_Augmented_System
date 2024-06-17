import os
import threading
import time
from collections import deque

import cv2
import numpy as np
from matplotlib import pyplot as plt
from openai import OpenAI
from pymilvus import MilvusClient
import requests
import clip
import torch
from PIL import Image

from viewer import hl2ss
from viewer import hl2ss_lnm
from pymilvus import connections, db, MilvusClient, Milvus, model

class VideoComponent():

    # loading the needed data, clip model are used here

    def __init__(self, ip, mainSys):
        self._ip = ip
        # 配置 gpt3.5 并获取 API 键
        self.llm = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        # Load the model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_version = "ViT-B/32"
        self.clip_feat_dim = \
            {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768, 'RN50x64': 1024, 'ViT-B/32': 512,
             'ViT-B/16': 512,
             'ViT-L/14': 768}[clip_version]
        self.model, self.preprocess = clip.load(clip_version)  # clip.available_models()
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
        self.mainSys = mainSys
        # 5 newest frames
        self.frame_queue = deque(maxlen=5)


    # Below are the necessary functions for clip model
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


    def get_description(self, img):
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


    # This is used to communicate with hololens using hl2ss
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



    # The method for the operating thread
    def process_data(self):
        while True:
            if self.frame_queue:
                img = self.frame_queue[-1]  # 取出最新的一帧进行处理
                thought = self.get_description(img)
                self.mainSys.read_then_write(thought)


    # The method for the steam data thread
    def video_component_on_invoke(self):
        thread_created = False
        client = self.get_video_client()
        client.open()

        while True:
            video_data = client.get_next_packet()
            cv2.imshow('Video', video_data.payload.image)
            self.frame_queue.append(video_data.payload.image)
            # Creat a new thread to call asynchronous function
            if not thread_created:
                thread = threading.Thread(target=self.process_data)
                thread.start()
                thread_created = True
            cv2.waitKey(1)









