import os
import shutil
import cv2
from annotator.canny import CannyDetector
from annotator.util import resize_image, HWC3

# 입력 데이터셋 폴더
input_dir = "/home/work/Team-RCD/png"  # 기존 폴더 (bear, frog, pig 등 클래스 폴더가 있음)
output_dir = "./dataset"  # 변환 후의 폴더
images_dir = os.path.join(output_dir, "images")
controls_dir = os.path.join(output_dir, "controls")
captions_file = os.path.join(output_dir, "captions.txt")

# 필요한 폴더 생성
os.makedirs(images_dir, exist_ok=True)
os.makedirs(controls_dir, exist_ok=True)

# Canny Edge Detector 초기화
canny = CannyDetector()

# 각 클래스별 프롬프트 정의
class_prompts = {
    "bear": "A bear",
    "bee" : "A bee",
    "bird" : "A bird",
    "butterfly" : "A butterfly",
    "camel" : "A camel",
    "cat" : "A cat",
    "cow" : "A cow",
    "crab" : "A crab",
    "crocodile" : "A crocodile",
    "dog" : "A dog",
    "dolphin" : "A dolphin",
    "dragon" : "A dragon",
    "duck" : "A duck",
    "elephant" : "A elephant",
    "fish" : "A fish",
    "frog": "A fog",
    "giraffe" : "A giraffe",
    "hedgehog" : "A real hedgehog",
    "horse" : "A horse",
    "kangaroo" : "A kangaroo",
    "lion" : "A lion",
    "monkey" : "A monkey",
    "octopus" : "A octopus",
    "owl" : "A owl",
    "pig" : "A pig",
    "rabbit" : "A rabbit",
    "sea turtle" : "A sea turtle",
    "shark" : "A shark",
    "sheep" : "A sheep",
    "snail" : "A snail",
    "snake" : "A snake",
    "squirrel" : "A squirrel",
    "tiger" : "A tiger",
    
}

# 캡션 파일 열기
with open(captions_file, "w") as caption_f:
    # 클래스 폴더 순회
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue  # 폴더가 아니면 무시

        # 각 클래스에 대한 프롬프트 설정
        prompt = class_prompts.get(class_name, "A beautiful animal")  # 기본값 설정

        # 이미지 파일 순회
        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            if not image_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue  # 이미지가 아니면 무시

            # 새 파일명 생성
            base_name = f"{class_name}_{os.path.splitext(image_file)[0]}"
            new_image_name = f"{base_name}.png"
            new_control_name = f"{base_name}_control.png"

            # 이미지 복사
            new_image_path = os.path.join(images_dir, new_image_name)
            shutil.copy(image_path, new_image_path)

            # Control Map 생성
            image = cv2.imread(image_path)
            image = HWC3(image)
            control_map = canny(resize_image(image, 512), 100, 200)  # Low and High Threshold 설정
            new_control_path = os.path.join(controls_dir, new_control_name)
            cv2.imwrite(new_control_path, control_map)

            # 캡션 작성 (프롬프트 포함)
            caption_f.write(f"{new_image_name}\t{prompt}\n")

print("폴더 변환이 완료되었습니다!")
