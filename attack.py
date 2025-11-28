import copy
import math
import torch.nn as nn
from torchvision.transforms import functional as F1
import gc
import string
import argparse
import torch
import numpy as np
import random
from utils import logger
from clip_text import clip
from collections import OrderedDict
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers import StableDiffusionPipelineSafe
from transformers import CLIPFeatureExtractor
from transformers import BertTokenizer, BertForMaskedLM,BertLMHeadModel,T5Tokenizer, T5ForConditionalGeneration
import time
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import os
import torch, gc
from PIL import Image
import pickle

torch.backends.cudnn.enabled = False
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
def load_prompts(file_path):
    if file_path.endswith("p"):
        res = torch.FloatTensor(pickle.load(open(file_path, 'rb'))).to("cuda")
    elif file_path.endswith("pt"):
        res = torch.load(open(file_path, 'rb')).to("cuda").to(torch.float32)
    return res
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
class SimClassifier(torch.nn.Module):
    def __init__(self, embeddings):
        super(SimClassifier, self).__init__()
        self.prompts = torch.nn.Parameter(embeddings)
        self.text_projection = nn.Linear(768, 512)
            
    def forward(self, x, clipmodel):
        text_feat = self.prompts / self.prompts.norm(dim=-1, keepdim=True)
        text_feat = self.text_projection(text_feat)
        text_feat = text_feat.to(dtype=torch.float16)

        image_feat = clipmodel.encode_image(x)
        # Pick the top 5 most similar labels for the image
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_feat @ text_feat.T)
        return similarity.squeeze()
    
def Q16_check(checkpoint):

    soft_prompts = load_prompts(checkpoint)
    classifier = SimClassifier(soft_prompts)
    classifier.to("cuda")
    return classifier

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# 创建黑图
def create_black_image(width, height):
    return Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))

class TextEncoder(nn.Module):
    def __init__(self, args, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = nn.Linear(512, 768)
        self.text_projection.half()
        self.dtype = clip_model.dtype
        self.dropout = nn.Dropout(0.5)
        self.n_ctx = args.n_ctx

    def forward(self, prompt_prefix, e_sen, weight, flag=False):

        x = prompt_prefix + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  
        if flag:
            x = self.transformer(x)
        else:
            counter=0
            outputs = self.transformer.resblocks([x,e_sen,weight,counter])
            x = outputs[0]

        x = x.permute(1, 0, 2)  
        x = self.ln_final(x).type(self.dtype)
        x = self.dropout(x)

        x = self.text_projection(x)
        x = x[:, :self.n_ctx]
        return x


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
class CustomRandomResizedCrop:
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.), interpolation=InterpolationMode.BILINEAR, antialias=True):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.antialias = antialias

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = F1.crop(img, i, j, h, w)
        return F1.resize(img, self.size, self.interpolation, antialias=self.antialias)

    @staticmethod
    def get_params(img, scale, ratio):
        width, height = img.size
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

class SDattack(object):
  """ A class used to manage adversarial prompt attacks. """
  
  def __init__(self,args, clipmodel, tokenizer, model, nsfw_list):
      self.tokenizer = tokenizer
      self.model = model
      self.model = self.model.to("cuda")

      vis_dim = clipmodel.visual.output_dim
      ctx_dim = clipmodel.ln_final.weight.shape[0]
      self.meta_net = nn.Sequential(
          OrderedDict([("linear1", nn.Linear(vis_dim, vis_dim // 4, bias=True)),
                       ("relu", QuickGELU()),
                       ("linear2", nn.Linear(vis_dim // 4, 4 * ctx_dim, bias=True))
                       ]))
      if args.prec == "fp16":
          self.meta_net.half()
      self.meta_net = self.meta_net.to("cuda")

      for param in self.model.parameters():
          param.requires_grad = False
      self.prompt_encoder = TextEncoder(args, clipmodel)
      self.prompt_encoder = self.prompt_encoder.to("cuda")
      self.nsfw_word_list = nsfw_list

  def generat_random_string(self, n):
      words = []
      for _ in range(n):
          word = ''.join(random.choices(string.ascii_letters, k=1))
          words.append(word)
      sentence = ' '.join(words)
      return sentence
  def prepare_image(self, img):
      image = np.array(img.convert("RGB"))
      image = image[None].transpose(0, 3, 1, 2)
      image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0  # [1,3,512,512]
      return image

  def get_image_features(self, image):
      X_adv = self.prepare_image(image)
      resize = torch.nn.functional.interpolate(X_adv, size=(224, 224), mode='bilinear',
                                               align_corners=False)
      normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      image_tensor = normalize(resize[0]) 
      image_tensor = image_tensor.unsqueeze(0) 

      return  image_tensor

  def calculate_loss(self, image, clip_model, target_text_feature, surr_image_features):
        """计算图像的损失"""
        image_tensor = self.get_image_features(image)
        image_tensor = image_tensor.to("cuda")
        image_features = clip_model.visual(image_tensor.type(clip_model.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
        score_1 = cos(image_features, target_text_feature)
        score_2 = cos(image_features, surr_image_features)
        
        loss1 = 1 - torch.mean(score_1)
        loss2 = 1 - torch.mean(score_2)
        loss_ms = loss1 + loss2
        
        return loss_ms.item()

  def compute_loss_ms(self, prompt_encoder_params, clip_model, pipe, safety_checker, 
                      target_text_feature, surr_image_features, args, tokenizer, input_ids, e_sen,
                      use_gaussian_noise_for_black=False, step_index=None):
        """计算L_MS损失函数"""
        with torch.no_grad():
            current_params = {}
            for name, param in self.prompt_encoder.named_parameters():
                current_params[name] = param.clone()
            
            for name, param in self.prompt_encoder.named_parameters():
                param.data = prompt_encoder_params[name]
            
            new_prompt_str = self.generat_random_string(args.n_ctx)
            new_prompt_tokens = clip.tokenize(new_prompt_str).to("cuda")
            new_prompt_ebd = clip_model.encode_text(new_prompt_tokens)
            new_prompt_ebd = new_prompt_ebd / new_prompt_ebd.norm(dim=-1, keepdim=True)
            new_prompt_ebd = new_prompt_ebd.unsqueeze(1).expand(-1, 77, -1)
            
            try:
                e_pe = self.prompt_encoder(new_prompt_ebd, e_sen, args.weight)
                e_pe = torch.where(torch.isnan(e_pe), torch.zeros_like(e_pe), e_pe)
                e_pe = torch.where(torch.isinf(e_pe), torch.zeros_like(e_pe), e_pe)
                
                text_embeds = self.model.bert.embeddings.word_embeddings(input_ids)
                inputs_embeds = torch.cat([e_pe, text_embeds], dim=1)
                outputs = self.model(inputs_embeds=inputs_embeds)
                logits = outputs.logits
                predicted_token_ids = logits.argmax(axis=-1)
                result_sentence = self.tokenizer.decode(predicted_token_ids[0])


                contains_nsfw = False
                if hasattr(self, 'nsfw_word_list') and self.nsfw_word_list:
                    result_sentence_lower = result_sentence.lower()
                    for nsfw_word in self.nsfw_word_list:
                        if nsfw_word.lower() in result_sentence_lower:
                            contains_nsfw = True
                if contains_nsfw:
                    black_image = create_black_image(512, 512) 
                    for name, param in self.prompt_encoder.named_parameters():
                        param.data = current_params[name] 
                    return black_image, True, result_sentence  
            except Exception as e:
                print(f"Error in prompt encoding: {e}")
                return float('inf'), True, "<pad>" * 20
            
            if args.safety_checker == "sc":
                with torch.autocast("cuda"):
                    image = pipe(result_sentence, num_inference_steps=50).images[0]
            elif args.safety_checker == "q16":
                with torch.autocast("cuda"):
                    image = pipe(result_sentence, num_inference_steps=50).images[0]
                images = [preprocess(image)]
                images = torch.stack(images).to("cuda")
                y = safety_checker(images, clip_model)
                y = torch.argmax(y, dim=0).detach().cpu().numpy()
                if y == 1:
                    image = create_black_image(image.width, image.height)
            
            is_black_image = self.is_black_image(image)
            
            if is_black_image and use_gaussian_noise_for_black and step_index == 0:
                image = self.generate_gaussian_noise_image(image.width, image.height)
                is_black_image = False
            for name, param in self.prompt_encoder.named_parameters():
                param.data = current_params[name]
        
        return image, is_black_image, result_sentence
  def is_black_image(self, image):
        """检查图像是否为黑图"""
        img_array = np.array(image)
        return np.all(img_array == 0)
  
  def generate_gaussian_noise_image(self, width, height):
        """生成高斯噪声图像用于restart策略"""
        noise = np.random.randn(height, width, 3) * 127.5 + 127.5
        noise = np.clip(noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noise)
  
  def estimate_zoo_gradient(self, current_params, clip_model, pipe, safety_checker,
                             target_text_feature, surr_image_features, args, tokenizer, input_ids, e_sen,
                             prev_gradient=None, c=0.01, beta=0.9, eta=0.001, step_index=None):
        """估计ZOO梯度"""
        delta = {}
        for name, param in current_params.items():
            delta[name] = torch.randn_like(param) * 0.01 
        
        params_plus = {}
        for name, param in current_params.items():
            params_plus[name] = param + c * delta[name]
        
        image_plus, is_black_plus, sen_plus = self.compute_loss_ms(
        params_plus, clip_model, pipe, safety_checker,
        target_text_feature, surr_image_features, args, tokenizer, input_ids, e_sen,
        use_gaussian_noise_for_black=False,
        step_index=step_index
    )
        
        params_minus = {}
        for name, param in current_params.items():
            params_minus[name] = param - c * delta[name]
        
        image_minus, is_black_minus, sen_minus = self.compute_loss_ms(
        params_minus, clip_model, pipe, safety_checker,
        target_text_feature, surr_image_features, args, tokenizer, input_ids, e_sen,
        use_gaussian_noise_for_black=False,
        step_index=step_index)
        
        g1 = {}
        both_black = is_black_plus and is_black_minus

        if step_index > 0 and both_black and prev_gradient is not None:
            print(f"Step {step_index}: Both perturbations produced black images, using historical gradient")
            return prev_gradient
  
        if step_index == 0 and both_black:
            print(f"Step 0: Both perturbations produced black images, using Gaussian noise for loss calculation")
            image_plus, _, _ = self.compute_loss_ms(
                params_plus, clip_model, pipe, safety_checker,
                target_text_feature, surr_image_features, args, tokenizer, input_ids, e_sen,
                use_gaussian_noise_for_black=True,
                step_index=step_index
            )
            
            image_minus, _, _ = self.compute_loss_ms(
                params_minus, clip_model, pipe, safety_checker,
                target_text_feature, surr_image_features, args, tokenizer, input_ids, e_sen,
                use_gaussian_noise_for_black=True,
                step_index=step_index
            )
            
            loss_plus = self.calculate_loss(image_plus, clip_model, target_text_feature, surr_image_features)
            loss_minus = self.calculate_loss(image_minus, clip_model, target_text_feature, surr_image_features)
            
            for name in current_params.keys():
                diff = loss_plus - loss_minus
                gradient = diff / (2 * c * delta[name] + 1e-8)
                gradient = torch.clamp(gradient, -1.0, 1.0)
                gradient = torch.where(torch.isnan(gradient), torch.zeros_like(gradient), gradient)
                gradient = torch.where(torch.isinf(gradient), torch.zeros_like(gradient), gradient)
                g1[name] = gradient
        else:
            loss_plus = self.calculate_loss(image_plus, clip_model, target_text_feature, surr_image_features)
            loss_minus = self.calculate_loss(image_minus, clip_model, target_text_feature, surr_image_features)
            
            for name in current_params.keys():
                diff = loss_plus - loss_minus
                gradient = diff / (2 * c * delta[name] + 1e-8)
                gradient = torch.clamp(gradient, -1.0, 1.0)
                gradient = torch.where(torch.isnan(gradient), torch.zeros_like(gradient), gradient)
                gradient = torch.where(torch.isinf(gradient), torch.zeros_like(gradient), gradient)
                g1[name] = gradient
        if prev_gradient is not None:
            g2 = {}
            for name in current_params.keys():
                g2[name] = beta * prev_gradient[name] + (1 - beta) * eta * g1[name]
                g2[name] = torch.where(torch.isnan(g2[name]), torch.zeros_like(g2[name]), g2[name])
                g2[name] = torch.where(torch.isinf(g2[name]), torch.zeros_like(g2[name]), g2[name])
                
            return g2
        else:
            return g1

  def run(self, args, target_nsfw_prompt, pipe, clip_model, surr_img, safety_checker):
    best_loss = float('inf')
    steps = 0

    tareget_prompt = clip.tokenize(target_nsfw_prompt)  
    tareget_prompt = tareget_prompt.to("cuda")
    with torch.no_grad():
        text_feature = clip_model.encode_text(tareget_prompt)
        text_feature = text_feature / text_feature.norm(dim=-1,
                                                        keepdim=True)

    e_tar = text_feature.to(torch.float16).to("cuda").detach()
    e_sen = self.meta_net(e_tar).reshape(e_tar.shape[0], -1, 512)

    input_ids = self.tokenizer.encode(target_nsfw_prompt, return_tensors='pt').to("cuda")
    current_params = {}
    for name, param in self.prompt_encoder.named_parameters():
        current_params[name] = param.clone().detach()
    surr_image_tensor = self.get_image_features(surr_img).to("cuda")
    surr_image_features = clip_model.visual(surr_image_tensor.type(clip_model.dtype))
    surr_image_features = surr_image_features / surr_image_features.norm(dim=-1, keepdim=True)
    
    prev_gradient = None

    for i in range(args.iter):
        steps += 1
        print(f"Iteration: {i}")
        
        estimated_gradient = self.estimate_zoo_gradient(
            current_params, clip_model, pipe, safety_checker,
            e_tar, surr_image_features, args, self.tokenizer, input_ids, e_sen,
            prev_gradient=prev_gradient, c=args.c, beta=args.beta, eta=args.learning_rate,
            step_index=i
        )
        for name, param in current_params.items():
            grad = estimated_gradient[name]
            grad = torch.where(torch.isnan(grad), torch.zeros_like(grad), grad)
            grad = torch.where(torch.isinf(grad), torch.zeros_like(grad), grad) 
            grad = torch.clamp(grad, -0.1, 0.1)
            current_params[name] = current_params[name] - args.learning_rate * grad
            current_params[name] = torch.where(torch.isnan(current_params[name]), 
                                             torch.zeros_like(current_params[name]), 
                                             current_params[name])
            current_params[name] = torch.where(torch.isinf(current_params[name]), 
                                             torch.zeros_like(current_params[name]), 
                                             current_params[name])
        
        with torch.no_grad():
            for name, param in self.prompt_encoder.named_parameters():
                param.copy_(current_params[name])
        
        image, is_black, result_sentence = self.compute_loss_ms(
            current_params, clip_model, pipe, safety_checker,
            e_tar, surr_image_features, args, self.tokenizer, input_ids, e_sen
        )
        current_loss = self.calculate_loss(image, clip_model, e_tar, surr_image_features)
        
        print("Resulting sentence:", result_sentence)
        print("Current loss:", current_loss)
        print("Is black image:", is_black)

        prev_gradient = estimated_gradient

        if current_loss < best_loss:
            best_loss = current_loss
            best_learnable_prompt = result_sentence  # [1,512]
            best_steps = steps
            print(f"New best at step {steps}: Loss={best_loss}, Prompt={best_learnable_prompt}")
        
        torch.cuda.empty_cache()
    return best_loss, best_learnable_prompt, best_steps

processor = transforms.Compose([
    CustomRandomResizedCrop((512, 512), antialias=True),
    transforms.ToTensor(),
])
def main(args):
    with open(args.dataset, 'r') as file:
        prompt_list = file.readlines()
    target_nsfw_prompts = [line.strip() for line in prompt_list]

    with open(args.prompt_filter, 'r') as file2:
        nsfw_word_list = file2.readlines()
    nsfw_word_list = [line.strip() for line in nsfw_word_list]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    clipmodel = torch.load('src/ViT-B-16.pt')
    clipmodel = clip.build_model(clipmodel.state_dict())
    clipmodel = clipmodel.to("cuda")
    sd_model = "runwayml/stable-diffusion-v1-5"
    if args.safety_checker == "sc":
        safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
        safety_checker = safety_checker.to("cuda")
        feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
        pipe = StableDiffusionPipelineSafe.from_pretrained(sd_model, safety_checker = safety_checker, feature_extractor = feature_extractor, 
                                                   torch_dtype=torch.float16)
    if args.safety_checker == "q16":
        safety_checker = Q16_check(checkpoint="src/prompts.p")
        pipe = StableDiffusionPipelineSafe.from_pretrained(sd_model,torch_dtype=torch.float16, safety_checker = safety_checker)

    pipe = pipe.to("cuda")
    clip_model = copy.deepcopy(clipmodel)
    clip_model = clip_model.to("cuda")
    logger.setup_logger(args.output_dir)

    surr_image_path = args.surrogate_image
    surr_imgs_file= [os.path.join(surr_image_path, f) for f in os.listdir(surr_image_path) if f.endswith('.png')]
    surr_imgs = [Image.open(f) for f in surr_imgs_file]
    advfile = open(args.gen_adv_prompt, 'a', encoding='utf-8')

    for i, target_nsfw_prompt in enumerate(target_nsfw_prompts):
        start = time.time()
        print(f"attack image id: {i}, prompt: {target_nsfw_prompt}")
        surr_img = surr_imgs[i]

        prompt_attack = SDattack(args = args, clipmodel=clipmodel, tokenizer = tokenizer, model = model, nsfw_list = nsfw_word_list) #初始化攻击类
        loss, learnable_prompt, steps = prompt_attack.run(args, target_nsfw_prompt, pipe, clip_model, surr_img, safety_checker) #执行攻击
        print("best_control:", learnable_prompt)
        print("best_loss:", loss)
        print("best_steps:", steps)
        runtime = time.time() - start
        print("time:", runtime)
        advfile.write(learnable_prompt + '\n')
        gc.collect()
        torch.cuda.empty_cache()
    
    advfile.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='attack diffusion.')
  parser.add_argument('-s','--random_seed', type=int, default=3 ,help='The random seed.')
  parser.add_argument('-i','--iter', type=int, default=200 ,help='iteration')
  parser.add_argument("--num_inference_steps", type=int, default=25)
  parser.add_argument("--eps", type=float, default=16.0)
  parser.add_argument("--weight", type=float, default=0.8)
  parser.add_argument("--n_ctx", type=int, default=6)
  parser.add_argument("--max_length", type=int, default=20, help="Maximum length of generated sentence")

  parser.add_argument(
      "--prec",
      type=str,
      default="fp16"
  )
  parser.add_argument(
      "--dataset",
      type=str,
      default='src/DATA/nudity.txt'
  )
  parser.add_argument(
      "--prompt_filter",
      type=str,
      default='src/DATA/nsfw_list.txt'
  )
  parser.add_argument("--output_dir",
                      type=str,
                      default="output",
                      help="output directory")
  parser.add_argument(
      "--gen_adv_prompt",
      type=str,
      default='src/DATA/gen_adv_prompt.txt',
      help="gen_adv_prompt"
  )
  parser.add_argument(
      "--surrogate_image",
      type=str,
      default='src/surrogate_model_image',
      help="surrogate_image"
  )
  parser.add_argument("--safety_checker",
                      type=str,
                      default="sc")
  parser.add_argument("--learning_rate",
                      type=float, default=0.0001)
  parser.add_argument("--beta",
                      type=float, default=0.8)
  parser.add_argument("--c",
                      type=float, default=0.0005)


  args = parser.parse_args()
  set_seed(args.random_seed)
  print(args)
  main(args)
