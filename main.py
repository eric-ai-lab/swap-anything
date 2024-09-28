import os
import sys
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm
import json
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import shutil
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from inversion import NullInversion
from swapping_class import AttentionSwap, LocalBlend
import utils
from utils import text2image
from utils import get_refinement_mapper

import cv2
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
from exist_config import fixed_config_list

import yaml
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Process configuration file.')
parser.add_argument('--config', type=str, required=True, help='Path to the config.yml file')

# Parse the arguments
args = parser.parse_args()

# Load the config.yml file
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

cuda_id = config['cuda_id']
do_not_crop = config['do_not_crop']
blend_width = config['blend_width']
total_diffusion_steps = config['total_diffusion_steps']

source_image_path = config['source_image_path']
source_subject_word = config['source_subject_word']
source_prompt = config['source_prompt']
target_subject_word = config['target_subject_word']
target_prompt = config['target_prompt']
concept_model_path = config['concept_model_path']

self_output_range = config['self_output_range']
self_map_range = config['self_map_range']
cross_map_range = config['cross_map_range']
add_zero_to_range = config['add_zero_to_range']

GUIDANCE_SCALE = config['guidance_scale']

end_blend = config['end_blend']

is_show_result = config['is_show_result']

pre_defined_crop = config['pre_defined_crop']

if (source_image_path.split('.')[0] + '.json') in fixed_config_list:
    config_path = source_image_path.split('.')[0] + '.json'
    print("Using existing config file.")
else: 
    config_path = None
    print("Generating new config file.")

if add_zero_to_range:
    range_combination_list = [(0.0, 0.0, 0.0)]
else:
    range_combination_list = []
for aa in self_output_range:
    for bb in self_map_range:
        for cc in cross_map_range:
            range_combination_list.append((aa, bb, cc))


def getROI(mask, margin_ratio, thres=0.25):
        mask = cv2.threshold(img_as_ubyte(mask, force_copy=True), thres, 1, cv2.THRESH_BINARY)[1]
        non_zero_point = cv2.findNonZero(mask)
        if non_zero_point is not None:
            x11, y11, x21, y21 = cv2.minMaxLoc(non_zero_point[:, :, 0])[0], cv2.minMaxLoc(non_zero_point[:, :, 1])[0], \
                                 cv2.minMaxLoc(non_zero_point[:, :, 0])[1], cv2.minMaxLoc(non_zero_point[:, :, 1])[1]

        else:
            x21 = mask.shape[1]
            x11 = 0
            y21 = mask.shape[0]
            y11 = 0

        margin = int(margin_ratio * min(x21 - x11, y21 - y11))
        if margin > 0:
            x11 = max(0, x11 - margin)
            x21 = min(mask.shape[1], x21 + margin)
            y11 = max(0, y11 - margin)
            y21 = min(mask.shape[0], y21 + margin)
        # fixme: here should we use None?
        if (x11 >= x21) or (y11 >= y21):
            x21 = mask.shape[1]
            x11 = 0
            y21 = mask.shape[0]
            y11 = 0

        return (int(x11), int(y11), int(x21), int(y21))


def convert_image_to_binary_mask(path):
    from PIL import Image
    import numpy as np
    
    # Read the image
    image = Image.open(path).convert("L")
    image_np = np.array(image)
    
    # Create binary mask
    binary_mask = np.where(image_np > 127, 1, 0).astype('uint8')
    
    return binary_mask


class Config:

    def __init__(self, image_json_path=None):
        
        with open(image_json_path, 'r') as f:
            image_dict = json.load(f)
        self.image_dict = image_dict
        self.source_image_path = image_dict['square_source_image_path']
        self.source_prompt = image_dict['source_prompt']
        self.source_subject_word = image_dict['source_subject_word']
        self.target_prompt = image_dict['target_prompt']
        self.target_subject_word = image_dict['target_subject_word']  
        self.concept_model_path = image_dict['concept_model_path']
        self.external_mask_path = image_dict['square_source_mask_path']
        
        self.cross_attention_map = 0.0
        self.self_attention_output = 0.0
        self.use_external_mask = False
        self.use_mask_on_self_attention = True
        self.area_mask_soft = 0.0
        self.time_step_soft = 0.0

        self.external_mask = None

        self.seed = 1

        self.change_distribution_after_swap = False
        self.note = 'face'
        
        
    def to_json(self):
        # Create a copy of the object's dictionary
        data = self.__dict__.copy()
        
        # Remove the 'mask' key from the dictionary if it exists
        if 'external_mask' in data:
            del data['external_mask']
        
        # Convert the modified dictionary to a JSON string
        return json.dumps(data)


    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        instance = cls()
        instance.__dict__.update(data)
        return instance


def apply_soft_mask(mask):    

    # Create a dilation kernel. You can change its size for different effects.
    kernel_size = 10
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Dilate the mask
    soft_mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Soften the edges using a Gaussian blur
    soft_mask = cv2.GaussianBlur(soft_mask.astype(np.float32), (25, 25), 0)
    
    # Restore the original 1 values
    soft_mask[mask == 1] = 1
    
    return soft_mask


def masked_replace_with_rescale(A, B, mask):
    
    # import pdb; pdb.set_trace()
    mean_A = A[mask].mean()
    std_A = A[mask].std()
    mean_B = B[mask].mean()
    std_B = B[mask].std()
    
    A_normalized = (A - mean_A) / (std_A)
    A_rescaled = A_normalized * std_B + mean_B
    A[mask] = A_rescaled[mask]
    # B_normalized = (B - mean_B) / (std_B)
    # B_rescaled = B_normalized * std_A + mean_A
    # A[mask] = B_rescaled[mask]
    return A

def generate_json_config(source_image_path, source_subject_word, source_prompt, target_subject_word, target_prompt, concept_model_path):
    
    dict = {}

    source_image_path = source_image_path
    dict['source_subject_word'] = source_subject_word
    dict['source_prompt'] = source_prompt
    dict['target_subject_word'] = target_subject_word
    dict['target_prompt'] = target_prompt
    dict['concept_model_path'] = concept_model_path
    
    dict['source_image_path'] = source_image_path
    dict['source_mask_path'] = source_image_path.split('.')[0] + '_mask.png'
    dict['square_source_image_path'] = source_image_path
    
    index = dict['source_image_path'].rfind('/')
    dict['square_source_image_path'] = dict['source_image_path'][:index+1] + 'square_' + dict['source_image_path'][index+1:]
    
    index = dict['source_mask_path'].rfind('/')
    dict['square_source_mask_path'] = dict['source_mask_path'][:index+1] + 'square_' + dict['source_mask_path'][index+1:]
    
    mask = cv2.imread(dict['source_mask_path'])[:,:,0] / 255.0
    domo_mask_binary = np.zeros_like(mask)
    domo_mask_binary[mask > 0.5] = 1
    domo_mask_binary[mask <= 0.5] = 0
    mask = domo_mask_binary
    
    x1,y1,x2,y2 = getROI(mask,0.2)
    mask = mask * 255.0

    if do_not_crop:
        x1 = 0
        y1 = 0
        x2 = mask.shape[1]
        y2 = mask.shape[0]
    if pre_defined_crop:
        x1, y1, x2, y2 = pre_defined_crop
    
    dict['crop_area'] = [x1, y1, x2, y2]
    
    new_mask = mask[y1:y2,x1:x2]
    
    new_mask = cv2.resize(new_mask,(512,512),cv2.INTER_NEAREST)  
    plt.imshow(new_mask)
    
    cv2.imwrite(dict['square_source_mask_path'],new_mask)
    
    img = cv2.imread(dict['source_image_path'])
    
    new_img = img[y1:y2,x1:x2]
    new_img = cv2.resize(new_img,(512,512),cv2.INTER_LINEAR)  
    plt.imshow(new_img[:,:,::-1])
    cv2.imwrite(dict["square_source_image_path"],new_img)
    
    dict['json_path'] = dict['source_image_path'].split('.')[0] + '.json'
    
    if os.path.exists(dict['json_path']):
        print("The file exists.")
        with open(dict['json_path'], 'w') as f:
            json.dump(dict, f)
    else:
        with open(dict['json_path'], 'w') as f:
            json.dump(dict, f)

    print(dict['json_path'])
    return dict['json_path']


if config_path is None:
    config_path = generate_json_config(source_image_path, source_subject_word, source_prompt, target_subject_word, target_prompt, concept_model_path)


class LocalBlend:
    
    def get_mask(self, x_t, maps, alpha, use_pool):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        return mask
    
    def __call__(self, x_t, attention_store):
        self.counter += 1

        self.start_blend = -1
        self.end_blend = end_blend
        if self.counter > self.start_blend and self.counter < self.end_blend:
           
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, 77) for item in maps]
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(x_t, maps, self.alpha_layers, True)
            mask = mask[:1] + mask

            import pdb
            import pickle
            # if self.counter<47:
            # mask_add = cv2.imread(config.external_mask_path)[:,:,0] / 255.0 #convert_image_to_binary_mask(config.external_mask_path)
            # mask_add = cv2.resize(mask_add, (512, 512), interpolation=cv2.INTER_NEAREST)   
            mask_64 = config.external_mask[3::8, 3::8]
            
            if config.area_mask_soft != 0.0:     
                mask_64 = apply_soft_mask(mask_64.astype(np.float32))
            
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()

            if config.use_external_mask:
                new_mask = torch.ones_like(mask)
                new_mask[:,:] = torch.tensor(mask_64).cuda()
            else:
                new_mask = mask.clone().detach().to(mask.device)

            if config.time_step_soft !=0 and config.use_external_mask:
                rate = self.counter * 1.0 / total_diffusion_steps * config.time_step_soft
                rate = min(rate, 1.0)
                new_mask = new_mask * rate
                 
            def masked_replace_with_rescale(A, B, mask):
                
                # import pdb; pdb.set_trace()
                mean_A = A[mask].mean()
                std_A = A[mask].std()
                mean_B = B[mask].mean()
                std_B = B[mask].std()
                
                A_normalized = (A - mean_A) / (std_A)
                A_rescaled = A_normalized * std_B + mean_B
                A[mask] = A_rescaled[mask]

                return A
    
            if self.counter > -1 and config.latent_change_distribution:

                x_t[1] = masked_replace_with_rescale(x_t[1], x_t[0], new_mask[0].repeat(4, 1, 1).bool())
                x_t = x_t[:1] + new_mask * (x_t - x_t[:1])
            else:
                x_t = x_t[:1] + new_mask * (x_t - x_t[:1])
        if self.counter < config.start_with_same_latent:
            x_t[1] = x_t[0]
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], tokenizer, device, NUM_DDIM_STEPS,
                 substruct_words=None, start_blend=0.2, th=(.3, .3), start_with_same_latent=0):
        self.start_with_same_latent = start_with_same_latent
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, 77)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            # import pdb; pdb.set_trace()
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        
        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, 77)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = utils.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * NUM_DDIM_STEPS)
        self.counter = 0 
        self.th=th
        

class AttentionControlEdit(abc.ABC):
    
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if self.LOW_RESOURCE else 0

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        # import pdb; pdb.set_trace()
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if self.LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        # import pdb;pdb.set_trace()
        if self.use_mask_on_attention:
            # import pdb;pdb.set_trace()
            
            def majority_pool(array, target_length):
                # Calculate the segment size
                segment_size = len(array) // target_length
                compressed_array = np.zeros(target_length, dtype=bool)
            
                for i in range(target_length):
                    segment = array[i * segment_size: (i + 1) * segment_size]
                    # Assign the mode of the segment
                    compressed_array[i] = np.sum(segment) > segment_size / 2
            
                return compressed_array
            
            compressed_mask = majority_pool(self.mask.reshape(-1), attn_base.shape[1])
            if att_replace.shape[2] <= 32 ** 2:
                self_attention_mask = torch.ones_like(attn_base[0])
                self_attention_mask[compressed_mask, :] = 0
                self_attention_mask[:, compressed_mask] = 0
                if self.attention_change_distribution:
                    att_replace_ = masked_replace_with_rescale(att_replace, attn_base.unsqueeze(0), self_attention_mask.bool().unsqueeze(0).unsqueeze(0).repeat(1, attn_base.shape[0],1,1))
                    return attn_base * self_attention_mask  + att_replace_ * (1-self_attention_mask)
                return attn_base * self_attention_mask  + att_replace * (1-self_attention_mask)
            else:
                return att_replace            
        else:
            if att_replace.shape[2] <= 32 ** 2:
                attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
                return attn_base
            else:
                return att_replace

            
    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0
        
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):

        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:
            self.step_store[key].append(attn)
            
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend], tokenizer, device):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        
        self.batch_size = len(prompts)
        self.cross_replace_alpha = utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend
        
        
class AttentionSwap(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_map_replace_steps: float, self_map_replace_steps: float, self_output_replace_steps: float,
                 source_subject_word=None, target_subject_word=None, tokenizer=None, device=None, LOW_RESOURCE=True, use_local_blend=True, mask=None,
                 use_mask_on_attention=None, use_mask_on_latent=None, attention_change_distribution=None, latent_change_distribution=None,
                 start_with_same_latent=0):
        self_map_replace_steps = self_map_replace_steps + self_output_replace_steps

        if use_local_blend:
            blend_word = (((source_subject_word,), (target_subject_word,)))
            local_blend = LocalBlend(prompts, blend_word, tokenizer, device, num_steps, start_with_same_latent=start_with_same_latent)
        else:
            local_blend = None
        
        super(AttentionSwap, self).__init__(prompts, num_steps, cross_map_replace_steps, self_map_replace_steps, local_blend, tokenizer, device)
        self.cross_map_replace_steps = cross_map_replace_steps
        self.self_map_replace_steps = self_map_replace_steps
        self.self_output_replace_steps = self_output_replace_steps
        self.mapper, alphas = get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])
        self.LOW_RESOURCE = LOW_RESOURCE
        self.mask = mask
        self.use_mask_on_attention = use_mask_on_attention
        self.use_mask_on_latent = use_mask_on_latent
        self.latent_change_distribution = latent_change_distribution
        self.attention_change_distribution = attention_change_distribution



import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from tqdm import tqdm
from einops import rearrange, repeat


class ScoreParams:

    def __init__(self, gap, match, mismatch):
        self.gap = gap
        self.match = match
        self.mismatch = mismatch

    def mis_match_char(self, x, y):
        if x != y:
            return self.mismatch
        else:
            return self.match


def get_matrix(size_x, size_y, gap):
    matrix = np.zeros((size_x + 1, size_y + 1), dtype=np.int32)
    matrix[0, 1:] = (np.arange(size_y) + 1) * gap
    matrix[1:, 0] = (np.arange(size_x) + 1) * gap
    return matrix


def get_traceback_matrix(size_x, size_y):
    matrix = np.zeros((size_x + 1, size_y +1), dtype=np.int32)
    matrix[0, 1:] = 1
    matrix[1:, 0] = 2
    matrix[0, 0] = 4
    return matrix


def global_align(x, y, score):
    matrix = get_matrix(len(x), len(y), score.gap)
    trace_back = get_traceback_matrix(len(x), len(y))
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            left = matrix[i, j - 1] + score.gap
            up = matrix[i - 1, j] + score.gap
            diag = matrix[i - 1, j - 1] + score.mis_match_char(x[i - 1], y[j - 1])
            matrix[i, j] = max(left, up, diag)
            if matrix[i, j] == left:
                trace_back[i, j] = 1
            elif matrix[i, j] == up:
                trace_back[i, j] = 2
            else:
                trace_back[i, j] = 3
    return matrix, trace_back


def get_aligned_sequences(x, y, trace_back):
    x_seq = []
    y_seq = []
    i = len(x)
    j = len(y)
    mapper_y_to_x = []
    while i > 0 or j > 0:
        if trace_back[i, j] == 3:
            x_seq.append(x[i-1])
            y_seq.append(y[j-1])
            i = i-1
            j = j-1
            mapper_y_to_x.append((j, i))
        elif trace_back[i][j] == 1:
            x_seq.append('-')
            y_seq.append(y[j-1])
            j = j-1
            mapper_y_to_x.append((j, -1))
        elif trace_back[i][j] == 2:
            x_seq.append(x[i-1])
            y_seq.append('-')
            i = i-1
        elif trace_back[i][j] == 4:
            break
    mapper_y_to_x.reverse()
    return x_seq, y_seq, torch.tensor(mapper_y_to_x, dtype=torch.int64)


def get_mapper(x: str, y: str, tokenizer, max_len=77):
    x_seq = tokenizer.encode(x)
    y_seq = tokenizer.encode(y)
    score = ScoreParams(0, 1, -1)
    matrix, trace_back = global_align(x_seq, y_seq, score)
    mapper_base = get_aligned_sequences(x_seq, y_seq, trace_back)[-1]
    alphas = torch.ones(max_len)
    alphas[: mapper_base.shape[0]] = mapper_base[:, 1].ne(-1).float()
    mapper = torch.zeros(max_len, dtype=torch.int64)
    mapper[:mapper_base.shape[0]] = mapper_base[:, 1]
    mapper[mapper_base.shape[0]:] = len(y_seq) + torch.arange(max_len - len(y_seq))
    return mapper, alphas


def get_refinement_mapper(prompts, tokenizer, max_len=77):
    x_seq = prompts[0]
    mappers, alphas = [], []
    for i in range(1, len(prompts)):
        mapper, alpha = get_mapper(x_seq, prompts[i], tokenizer, max_len)
        mappers.append(mapper)
        alphas.append(alpha)
    return torch.stack(mappers), torch.stack(alphas)


def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False):
    # import pdb; pdb.set_trace()
    if low_resource:
        # import pdb; pdb.set_trace()
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[:2])["sample"]
        # pdb.set_trace()
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[2:])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        # import pdb; pdb.set_trace()
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]

    import pdb
    # pdb.set_trace()
    latents = controller.step_callback(latents)
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


@torch.no_grad()
def text2image(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image',
    LOW_RESOURCE=False
):
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    if text_input.input_ids.shape[0] == 2:
        text_embeddings = torch.cat((model.text_encoder(text_input.input_ids[:1].to(model.device))[0], model.text_encoder(text_input.input_ids[1:].to(model.device))[0]), dim=0)
    else:
        text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=LOW_RESOURCE)
        
    if return_type == 'image':
        image = latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent


def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
        
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None,temb=None,):
            is_cross = encoder_hidden_states is not None
            
            residual = hidden_states

            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)
            
            if controller.cur_att_layer > controller.num_uncond_att_layers and query.shape[0]==2 and not is_cross and 0 <= controller.cur_step <= int(controller.self_output_replace_steps * 50):
                
                query[1, :, :] = query[0, :, :]
                key[1, :, :] = key[0, :, :]
                value[1, :, :] = value[0, :, :]
            
            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            attention_probs = self.get_attention_scores(query, key, attention_mask)
            attention_probs = controller(attention_probs, is_cross, place_in_unet)

            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = to_out(hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            return hidden_states
        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count

    
def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor]=None):
    if type(bounds) is float:
        bounds = 0, bounds

    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words






scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
LOW_RESOURCE = True
NUM_DDIM_STEPS = total_diffusion_steps
device = torch.device(f'cuda:{cuda_id}') if torch.cuda.is_available() else torch.device('cpu')

all_json_files = [config_path]

for image_json_path in all_json_files[:]:
    # try:
        config_no_mask = Config(image_json_path)
        config_no_mask.use_external_mask = False
        
        config_hard_mask = Config(image_json_path)
        config_hard_mask.use_external_mask = True
        config_hard_mask.area_mask_soft = 0.0
        config_hard_mask.time_step_soft = 0.0
        
        config_area_soft_mask = Config(image_json_path)
        config_area_soft_mask.use_external_mask = True
        config_area_soft_mask.area_mask_soft = 1.0
        config_area_soft_mask.time_step_soft = 0.0
        
        config_area_time_soft_mask = Config(image_json_path)
        config_area_time_soft_mask.use_external_mask = True
        config_area_time_soft_mask.area_mask_soft = 1.0
        config_area_time_soft_mask.time_step_soft = 1.0

    
        html_img_idx_list = []
        
        for idx, config in enumerate([config_hard_mask]):
            target_mask = cv2.imread(config.external_mask_path)[:,:,0] / 255.0
            # target_mask = target_mask_.reshape(-1)
            config.external_mask = target_mask
            for concept_model_path in [config.concept_model_path]:

                
                if idx == 0:
                    concept_model = StableDiffusionPipeline.from_pretrained(concept_model_path, scheduler=scheduler).to(device)
                    try:
                        concept_model.disable_xformers_memory_efficient_attention()
                    except AttributeError:
                        print("Attribute disable_xformers_memory_efficient_attention() is missing")
                    tokenizer = concept_model.tokenizer
                
                source_image_path = config.source_image_path
                source_subject_word = config.source_subject_word
                source_prompt = config.source_prompt # Add another word 'woman' so the source and target prompt have the same token length. This is not a necessary step.
                # source_prompt = 'a black and white photo ' + source_prompt
                
                target_subject_word = config.target_subject_word
                target_prompt = config.target_prompt # Changing the subject into the target. 'sks woman' is the name used when training this example concept learning model.
                # concept_model_path = '../diffusers/examples/dreambooth/dreambooth-15-cartoon-2/checkpoint-800/'
                # The three parameters used to change the tuning. Different concept learning model may have different parameter range.
                if idx == 0: 
                    null_inversion = NullInversion(concept_model, ddim_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE)
                    x_t, uncond_embeddings = null_inversion.invert(source_image_path, source_prompt)
                
                
                # for config.self_output_replace_steps in self_output_range:
                #     for config.self_map_replace_steps in self_map_range:
                #         for config.cross_map_replace_steps in cross_map_range:
                for config.self_output_replace_steps, config.self_map_replace_steps, config.cross_map_replace_steps in range_combination_list:
                            
                # for config.self_output_replace_steps, config.self_map_replace_steps in [[0.0, 0.2], [0.0, 0.8]]:
                            # config.cross_map_replace_steps = 0.0
                            self_output_replace_steps = config.self_output_replace_steps
                            cross_map_replace_steps = config.cross_map_replace_steps
                        
                            assert config.self_output_replace_steps + config.self_map_replace_steps <= 1.0
                        
                            # for seed in [33]:
                            for config.use_external_mask in [True]:
                                # x_t = torch.randn(
                                #         (1, 4, 64, 64),
                                #         generator=torch.Generator().manual_seed(seed)
                                #     )
                                # uncond_embeddings = None                    
                                prompts = [source_prompt, target_prompt]
                                
                                for config.use_mask_on_attention in [True]:
                                    config.use_mask_on_latent = config.use_mask_on_attention
                                    for config.attention_change_distribution in [True]:
                                        config.latent_change_distribution = config.attention_change_distribution
                                        for config.start_with_same_latent in [0]:
                                            controller = AttentionSwap(prompts, NUM_DDIM_STEPS, 
                                                                       cross_map_replace_steps=config.cross_map_replace_steps, self_map_replace_steps=config.self_map_replace_steps, self_output_replace_steps=config.self_output_replace_steps,
                                                                       source_subject_word=source_subject_word, target_subject_word=target_subject_word,
                                                                       tokenizer=concept_model.tokenizer, device=device, LOW_RESOURCE=LOW_RESOURCE, 
                                                                       mask=config.external_mask, 
                                                                       use_mask_on_attention=config.use_mask_on_attention, use_mask_on_latent=config.use_mask_on_latent,  
                                                                       attention_change_distribution=config.attention_change_distribution,
                                                                       latent_change_distribution=config.latent_change_distribution,
                                                                       start_with_same_latent=config.start_with_same_latent,
                                                                      )
                                            
                                            images, x_t = text2image(concept_model, prompts, controller, latent=x_t, num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE,
                                                                     generator=None, uncond_embeddings=uncond_embeddings, LOW_RESOURCE=LOW_RESOURCE)
                                            
                                            out_dir = "./real_output_" + str(cuda_id) + "/"
                                            
                                            os.makedirs(out_dir, exist_ok=True)
                                            count = len(os.listdir(out_dir))
                                            out_dir = os.path.join(out_dir, f"sample_{count}")
                                            os.makedirs(out_dir, exist_ok=True)
                
                                            html_img_idx_list.append(out_dir)
                        
                                            # Serialize the instance to a file
                                            with open(out_dir+"/config.json", "w") as f:
                                                f.write(config.to_json())
                                            
                                            if is_show_result:
                                                print('Original Image:', out_dir+'/source.png')
                                                display(Image.fromarray(images[0]))
                                                print('\nImage after subject swapping:', out_dir+'/result.png')
                                                display(Image.fromarray(images[1]))
                                                
                                            Image.fromarray(images[0]).save(out_dir+'/source.png')
                                            Image.fromarray(images[1]).save(out_dir+'/result.png')
                    
                                            large_source_img = Image.open(config.image_dict['source_image_path'])
                                            large_result_img = large_source_img.copy()
                                            mask_img = Image.open(config.image_dict['source_mask_path'])
                                            
                                            original_height, original_width = config.image_dict['crop_area'][3] - config.image_dict['crop_area'][1], config.image_dict['crop_area'][2] - config.image_dict['crop_area'][0]
                                            resized_back_img = Image.fromarray(images[1]).resize((original_width, original_height))
                                            
                                            large_result_img.paste(resized_back_img, (config.image_dict['crop_area'][0], config.image_dict['crop_area'][1]))

                                            # get smooth image
                                            from PIL import Image, ImageDraw

                                            # Define the area for the crop and the blending width (feathering edge width)
                                            crop_area = config.image_dict['crop_area']  # Assuming this is a tuple (x1, y1, x2, y2)
                                              # Width of the blending edge in pixels

                                            # Create a new mask for blending with a linear gradient at the edges
                                            mask_width, mask_height = crop_area[2] - crop_area[0], crop_area[3] - crop_area[1]
                                            blend_mask = Image.new('L', (mask_width, mask_height), 255)
                                            draw = ImageDraw.Draw(blend_mask)

                                            # Create a gradient from opaque to transparent for the blending
                                            for i in range(blend_width):
                                                # Calculate the gradient value
                                                gradient_value = int(255 * ((i) / blend_width))
                                                
                                                # Apply the gradient around the edges
                                                draw.rectangle([i, i, mask_width - i, mask_height - i], outline=gradient_value)

                                            # Resize the blend_mask to match the size of the resized_back_img, if necessary
                                            # If original_width and original_height match resized_back_img's size, this step can be skipped
                                            blend_mask = blend_mask.resize((original_width, original_height))

                                            # Create a copy of large_result_img to apply the blended paste, preserving the original
                                            smoothed_result_img = large_source_img.copy()
                                            
                                            # Use the blend_mask to smoothly blend resized_back_img onto smoothed_result_img
                                            smoothed_result_img.paste(resized_back_img, (crop_area[0], crop_area[1]), blend_mask)


                                            large_source_img.save(out_dir+'/large_source.png')
                                            large_result_img.save(out_dir+'/large_result.png')
                                            smoothed_result_img.save(out_dir+'/smoothed_result.png')
                                            mask_img.save(out_dir+'/external_mask.png')
                                            print(f'latent_change_distribution: {config.latent_change_distribution}')
                                            print(f'use_mask_on_self_attention: {config.use_mask_on_latent}')
        # generate html
        import json
        
        # html_path = "html/" + config.image_dict['source_image_path'].split('/')[-1].split('.')[0] + '.html'

        import os

        def generate_unique_filename(base_path, filename):
            counter = 1  # Start with 1 for the potential first duplicate
            file_name_without_extension, extension = os.path.splitext(filename)
            
            # Construct the full path
            full_path = os.path.join(base_path, filename)
        
            # Check if the file exists, and find a unique name
            while os.path.exists(full_path):
                # Update the filename with a counter
                new_filename = f"{file_name_without_extension}_{counter}{extension}"
                full_path = os.path.join(base_path, new_filename)
                counter += 1
        
            return full_path
        
        # Example usage
        base_path = 'html'  # Replace with your directory
        # filename = config.image_dict['source_image_path'].split('/')[-1].split('.')[0] + '.html'
        filename = config.image_dict['source_image_path'].split('/')[-3].split('_')[-1] + "_" + config.image_dict['source_image_path'].split('/')[-1].split('.')[0] + '.html'

        html_path = generate_unique_filename(base_path, filename)
        print("Unique html path:", html_path)

        html_file = open(html_path, "w")
        html_file.write("<html><head><h2>Subject Swap Results</h2></head>\n")
        html_file.write("<body>\n<table>\n")
        
        html_file.write("<style>\n")
        html_file.write("td.caption { max-width: 300px; word-wrap: break-word; }\n")  # Limiting width and enabling word-wrap
        html_file.write("</style>\n")
        
        for path in html_img_idx_list:
            source_path = '../' + path + '/source.png'
            result_path = '../' + path + '/result.png'
            large_source_path =  '../' + path + '/large_source.png'
            large_result_path = '../' + path + '/large_result.png'
            large_smoothed_result_path = '../' + path + '/smoothed_result.png'
            mask_path = '../' + path + '/external_mask.png'
            json_path = path + '/config.json'
            
            text = []
            with open(json_path, 'r') as f:
                config_str = f.read()
            # config_str = json.loads()
            # print(result_path)
            html_file.write("<tr>\n")
            html_file.write('<td class="caption">\n')  # Added class "caption" to this <td>
            html_file.write("<h5>" + config_str + ", " + path + "</h5>\n")
            html_file.write("</td>\n")
            html_file.write("<td>\n")
            html_file.write('<img width=512 src="' + source_path + '"/>\n')
            html_file.write('<img width=512 src="' + result_path + '"/>\n')
            html_file.write('<img width=512 src="' + large_source_path + '"/>\n')
            html_file.write('<img width=512 src="' + large_result_path + '"/>\n')
            html_file.write('<img width=512 src="' + large_smoothed_result_path + '"/>\n')
            html_file.write('<img width=512 src="' + mask_path + '"/>\n')
            html_file.write("</td>\n")
            html_file.write("</tr>\n")
        html_file.write("</table>\n</body>\n</html>")
        html_file.close()
    # except:
    #     with open('not_success_path.txt', 'a+') as f:
    #         f.write(image_json_path)
    #         f.write('\n')
        


        