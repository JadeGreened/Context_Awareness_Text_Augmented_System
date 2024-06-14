import requests
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import google.generativeai as genai
import os
import clip
import torch



genai.configure(api_key=os.getenv("GOOGLE_API_KEY"), transport='rest')
llm = genai.GenerativeModel(model_name="gemini-pro")


# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_version = "ViT-B/32"
clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768, 'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}[clip_version]
model, preprocess = clip.load(clip_version)  # clip.available_models()

def get_text_feats(in_text, batch_size=64):
  text_tokens = clip.tokenize(in_text).to(device)
  text_id = 0
  text_feats = np.zeros((len(in_text), clip_feat_dim), dtype=np.float32)
  while text_id < len(text_tokens):  # Batched inference.
    batch_size = min(len(in_text) - text_id, batch_size)
    text_batch = text_tokens[text_id:text_id+batch_size]
    with torch.no_grad():
      batch_feats = model.encode_text(text_batch).float()
    batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
    batch_feats = np.float32(batch_feats.cpu())
    text_feats[text_id:text_id+batch_size, :] = batch_feats
    text_id += batch_size
  return text_feats

def get_img_feats(img):
  img_pil = Image.fromarray(np.uint8(img))
  img_in = preprocess(img_pil)[None, ...]
  with torch.no_grad():
    img_feats = model.encode_image(img_in.cuda()).float()
  img_feats /= img_feats.norm(dim=-1, keepdim=True)
  img_feats = np.float32(img_feats.cpu())
  return img_feats

def get_nn_text(raw_texts, text_feats, img_feats):
  scores = text_feats @ img_feats.T
  scores = scores.squeeze()
  high_to_low_ids = np.argsort(scores).squeeze()[::-1]
  high_to_low_texts = [raw_texts[i] for i in high_to_low_ids]
  high_to_low_scores = np.sort(scores).squeeze()[::-1]
  return high_to_low_texts, high_to_low_scores

def download_file(url, filename):
    if not os.path.exists(filename):
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
        else:
            print(f"Failed to download the file: {url}")

def prompt_llm(prompt):
  prompt_parts = [prompt]
  return llm.generate_content(prompt_parts).text


# Load scene categories from Places365.
# if not os.path.exists('categories_places365.txt'):
#   download_file("https://raw.githubusercontent.com/zhoubolei/places_devkit/master/categories_places365.txt","categories_places365.txt")
place_categories = np.loadtxt('categories_places365.txt', dtype=str)
place_texts = []
for place in place_categories[:, 0]:
  place = place.split('/')[2:]
  if len(place) > 1:
    place = place[1] + ' ' + place[0]
  else:
    place = place[0]
  place = place.replace('_', ' ')
  place_texts.append(place)
place_feats = get_text_feats([f'Photo of a {p}.' for p in place_texts])


# Load object categories from Tencent ML Images.
# if not os.path.exists('dictionary_and_semantic_hierarchy.txt'):
#   download_file("https://raw.githubusercontent.com/Tencent/tencent-ml-images/master/data/dictionary_and_semantic_hierarchy.txt","dictionary_and_semantic_hierarchy.txt")
with open('dictionary_and_semantic_hierarchy.txt') as fid:
    object_categories = fid.readlines()
object_texts = []
for object_text in object_categories[1:]:
  object_text = object_text.strip()
  object_text = object_text.split('\t')[3]
  safe_list = ''
  for variant in object_text.split(','):
    text = variant.strip()
    safe_list += f'{text}, '
  safe_list = safe_list[:-2]
  if len(safe_list) > 0:
    object_texts.append(safe_list)
object_texts = [o for o in list(set(object_texts)) if o not in place_texts]  # Remove redundant categories.
object_feats = get_text_feats([f'Photo of a {o}.' for o in object_texts])




# Download image.
img_url = "https://github.com/rmokady/CLIP_prefix_caption/raw/main/Images/COCO_val2014_000000165547.jpg" #@param {type:"string"}
fname = 'demo_img.png'
with open(fname, 'wb') as f:
    f.write(requests.get(img_url).content)

verbose = True#@param {type:"boolean"}

# Load image.
img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
img_feats = get_img_feats(img)
plt.imshow(img); plt.show()

# Zero-shot VLM: classify image type.
img_types = ['photo', 'cartoon', 'sketch', 'painting']
img_types_feats = get_text_feats([f'This is a {t}.' for t in img_types])
sorted_img_types, img_type_scores = get_nn_text(img_types, img_types_feats, img_feats)
img_type = sorted_img_types[0]

# Zero-shot VLM: classify number of people.
ppl_texts = ['no people', 'people']
ppl_feats = get_text_feats([f'There are {p} in this photo.' for p in ppl_texts])
sorted_ppl_texts, ppl_scores = get_nn_text(ppl_texts, ppl_feats, img_feats)
ppl_result = sorted_ppl_texts[0]
if ppl_result == 'people':
  ppl_texts = ['is one person', 'are two people', 'are three people', 'are several people', 'are many people']
  ppl_feats = get_text_feats([f'There {p} in this photo.' for p in ppl_texts])
  sorted_ppl_texts, ppl_scores = get_nn_text(ppl_texts, ppl_feats, img_feats)
  ppl_result = sorted_ppl_texts[0]
else:
  ppl_result = f'are {ppl_result}'

# Zero-shot VLM: classify places.
place_topk = 3
place_feats = get_text_feats([f'Photo of a {p}.' for p in place_texts ])
sorted_places, places_scores = get_nn_text(place_texts, place_feats, img_feats)

# Zero-shot VLM: classify objects.
obj_topk = 10
sorted_obj_texts, obj_scores = get_nn_text(object_texts, object_feats, img_feats)
object_list = ''
for i in range(obj_topk):
  object_list += f'{sorted_obj_texts[i]}, '
object_list = object_list[:-2]

# Zero-shot LM: generate captions.
num_captions = 10
prompt = f'''I am an intelligent image captioning bot.
This image is a {img_type}. There {ppl_result}.
I think this photo was taken at a {sorted_places[0]}, {sorted_places[1]}, or {sorted_places[2]}.
I think there might be a {object_list} in this {img_type}.
A creative short caption I can generate to describe this image is:'''
caption_texts = []
for i in range(num_captions):
  caption_texts.append(prompt_llm(prompt))
# Zero-shot VLM: rank captions.
caption_feats = get_text_feats(caption_texts)
sorted_captions, caption_scores = get_nn_text(caption_texts, caption_feats, img_feats)
print(f'{sorted_captions[0]}\n')

if verbose:
  print(f'VLM: This image is a:')
  for img_type, score in zip(sorted_img_types, img_type_scores):
    print(f'{score:.4f} {img_type}')

  print(f'\nVLM: There:')
  for ppl_text, score in zip(sorted_ppl_texts, ppl_scores):
    print(f'{score:.4f} {ppl_text}')

  print(f'\nVLM: I think this photo was taken at a:')
  for place, score in zip(sorted_places[:place_topk], places_scores[:place_topk]):
    print(f'{score:.4f} {place}')

  print(f'\nVLM: I think there might be a:')
  for obj_text, score in zip(sorted_obj_texts[:obj_topk], obj_scores[:obj_topk]):
    print(f'{score:.4f} {obj_text}')

  print(f'\nLM generated captions ranked by VLM scores:')
  for caption, score in zip(sorted_captions, caption_scores):
    print(f'{score:.4f} {caption}')