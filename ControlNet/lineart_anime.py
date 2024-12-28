import os
import cv2
import torch
import random
import numpy as np
import einops
from annotator.util import resize_image, HWC3
from annotator.lineart_anime import LineartAnimeDetector  # Lineart Anime Detector 가져오기
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from pytorch_lightning import seed_everything

# 모델 초기화
model_name = 'control_v11p_sd15s2_lineart_anime'
model = create_model(f'/home/work/Team-RCD/please/ControlNet/models/control_v11p_sd15s2_lineart_anime.yaml').cpu()
model.load_state_dict(load_state_dict('/home/work/Team-RCD/please/ControlNet/models/Anything-v4.5-pruned-mergedVae.ckpt', location='cuda'), strict=False)
model.load_state_dict(load_state_dict(f'/home/work/Team-RCD/please/ControlNet/models/control_v11p_sd15s2_lineart_anime.pth', location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)

# 전역 변수 초기화
preprocessor = None  # CannyDetector가 할당될 변수

# 입력 이미지 경로
input_image_path = "/home/work/Team-RCD/please/ControlNet/models/input_image.png"  

# 입력 이미지 로드 및 전처리
input_image = cv2.imread(input_image_path)
input_image = HWC3(input_image)  # Ensure image has 3 channels

# 사용자 정의 함수
def process(det, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    global preprocessor  # 전역 변수로 선언

    # 프리프로세서 초기화
    if det == 'Lineart_anime':
        if not isinstance(preprocessor, LineartAnimeDetector):
            preprocessor = LineartAnimeDetector()  # Lineart Anime Detector 초기화

    with torch.no_grad():
        if det == 'None':
            detected_map = input_image.copy()
        else:
            if det == 'Lineart_anime':
                detected_map = preprocessor(resize_image(input_image, detect_resolution))  # Lineart Anime 처리
            else:
                raise ValueError(f"Unsupported detector: {det}")
            
            detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]

    # 결과 저장
    save_dir = "/home/work/Team-RCD/please/ControlNet/results"
    os.makedirs(save_dir, exist_ok=True)
    for i, result in enumerate(results):
        save_path = os.path.join(save_dir, f"anime_output{i}.png")
        cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"Result saved at {save_path}")

# 파라미터 설정
det = 'Lineart_anime'
prompt = "A teddy bear, brown"  # 주요 프롬프트
a_prompt = "best quality"  # 추가 프롬프트
n_prompt = "lowres, bad anatomy, bad lighting"  # 네거티브 프롬프트
num_samples = 1
image_resolution = 512
detect_resolution = 512
ddim_steps = 20
guess_mode = False
strength = 1.0
scale = 9.0
seed = 12345
eta = 1.0
low_threshold = 100
high_threshold = 200

# 실행
process(det, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold)
