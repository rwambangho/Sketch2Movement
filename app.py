import streamlit as st
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from streamlit_drawable_canvas import st_canvas
import sys
import os
import torch
import cv2
import numpy as np
from ControlNet import test_lineart_anime
from photo_background_generation.background_generator import generate_background


# Add the directory containing test_lineart_anime.py to the Python path
sys.path.append('/home/work/Team-RCD/please/ControlNet')
sys.path.append('/home/work/Team-RCD/please/photo-background-generation')
# sys.path.append('/content/drive/MyDrive/ColabNotebooks/rcd/PIA')

# from test import *

# from rcd.PIA.inference import seed_everything

# 이미지 경로
input_image_path = "/home/work/Team-RCD/please/ControlNet/models/input_image.png"
controlnet_results_dir = "/home/work/Team-RCD/please/ControlNet/results"
background_results_dir = "/home/work/Team-RCD/please/photo-background-generation/results"
controlnet_result_path = os.path.join(controlnet_results_dir, "output_1.png")
background_result_path = os.path.join(background_results_dir, "background_output.png")

st.title("RCD Sketch AI")

# ---------------------------
# 이미지 업로드 섹션
st.header("Input Image File")
uploaded_image = st.file_uploader("Upload an image (png, jpg, jpeg)", type=["png", "jpg", "jpeg"])

# 이미지 열기 및 표시
bg_image = None
if uploaded_image is not None:
    bg_image = Image.open(uploaded_image)
    st.image(bg_image, caption="Uploaded Image", use_container_width=True)

# ---------------------------
# 캔버스 섹션
st.subheader("Draw Yourself!")

# 캔버스 설정 (사이드바)
stroke_width = st.sidebar.slider("선 두께 선택", 1, 25, 3)
stroke_color = st.sidebar.color_picker("선 색상 선택", "#000000")
bg_color = st.sidebar.color_picker("배경 색상 선택", "#FFFFFF")

# 캔버스 컴포넌트 설정
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",  # 채우기 색상 (투명)
    stroke_width=stroke_width,           # 선 두께
    stroke_color=stroke_color,           # 선 색상
    background_color=bg_color,           # 배경 색상
    height=400,                          # 캔버스 높이
    width=800,                           # 캔버스 너비
    drawing_mode="freedraw",             # 그리기 모드  
    key="canvas",
)


# ---------------------------
# 프롬프트 입력 섹션
st.subheader("Customize Your Prompt")
user_prompt = st.text_input("Enter your prompt", placeholder="프롬프트를 입력해주세요!")
user_a_prompt = "best quality"  
user_n_prompt = "lowres, bad anatomy, bad lighting"  


# ---------------------------
# Generate 버튼 섹션
if st.button("Generate Anime Style"):
    # Save input image
    os.makedirs(os.path.dirname(input_image_path), exist_ok=True)
    if uploaded_image:
        bg_image.save(input_image_path)
        st.success("Uploaded Image saved to models folder!")
        print(11)
    elif canvas_result.image_data is not None: #if draw
        sketch_image = Image.fromarray(canvas_result.image_data.astype('uint8')).convert("RGB")
        sketch_image.save(input_image_path)
    else:
        st.warning("Please upload an image or draw a sketch!")
        st.stop()

    # Load input image
    input_image = test_lineart_anime.load_image(input_image_path)
    if input_image is None:
        st.error("Image file could not be loaded. Please check the file.") 
        st.stop()

    #     # 모델 초기화 여부 확인
    # if test_lineart_anime.model is None:
    #     st.error("Model is not initialized. Please check the initialization process or model files.")
    #     st.stop()

    # Process AI Model
    try:
        torch.cuda.empty_cache()
        det = 'Lineart_anime'
        num_samples = 1
        image_resolution = 256
        detect_resolution = 256
        ddim_steps = 20
        guess_mode = False
        strength = 1.0
        scale = 9.0
        seed = 12345
        eta = 1.0
        low_threshold = 100
        high_threshold = 200

        test_lineart_anime.process(
            det, input_image, user_prompt, user_a_prompt, user_n_prompt,
            num_samples, image_resolution, detect_resolution,
            ddim_steps, guess_mode, strength, scale,
            seed, eta, low_threshold, high_threshold
        )

        # 결과 표시
        if os.path.exists(controlnet_result_path):
            st.image(controlnet_result_path, caption="ControlNet Output", use_container_width=True)
        else:
            st.error("ControlNet output not found.")
    except RuntimeError as e:
        st.error(f"RuntimeError: {str(e)}")
    finally:
        # GPU 메모리 정리
        if hasattr(test_lineart_anime, "model"):
            del test_lineart_anime.model
        if hasattr(test_lineart_anime, "ddim_sampler"):
            del test_lineart_anime.ddim_sampler
        if hasattr(test_lineart_anime, "preprocessor"):
            del test_lineart_anime.preprocessor
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------------------------
# Step 1: GPU memory cleanup
st.subheader("Step 1 Completed: Cleaning up GPU memory")
if 'test_lineart_anime' in locals():
    del test_lineart_anime.model
    del test_lineart_anime.ddim_sampler
    del test_lineart_anime.preprocessor
torch.cuda.empty_cache()
torch.cuda.synchronize()


# ---------------------------
# 두 번째 스텝: 배경 생성
st.header("Step 2: Background Generation")

# Check if ControlNet output exists
if not os.path.exists(controlnet_result_path):
    st.warning("ControlNet output not found. Complete Step 1 first.")
    st.stop()

# 배경 스타일 선택 및 선 굵기 조정 옵션
st.subheader("Choose a Background Style")
background_styles = ["Forest", "Beach", "Cityscape", "Mountains", "Jungle"]
background_prompt = st.selectbox("Select Background Style", background_styles)

adjust_thickness = st.radio("Enable Line Thickness Adjustment?", ["No", "Yes"], index=0) == "Yes"
kernel_size = 1
if adjust_thickness:
    kernel_size = st.slider("Set Line Thickness", 1, 15, 5)

# 배경 생성 실행 버튼
if st.button("Generate Background"):
    try:
        # GPU 메모리 정리
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # 배경 생성 호출
        generate_background(
            input_image_path=controlnet_result_path,  # Step 1 결과
            prompt=background_prompt,                # UI에서 선택한 배경 스타일
            output_image_path=background_result_path,  # 결과 저장 경로
            enable_dilation=adjust_thickness,         # 선 굵기 조정 여부
            kernel_size=kernel_size,                  # 커널 크기
        )

        # 결과 표시
        if os.path.exists(background_result_path):
            st.image(background_result_path, caption="Generated Background", use_container_width=True)
            st.success("Background generation complete!")
        else:
            st.error("Background generation failed.")
    except RuntimeError as e:
        st.error(f"RuntimeError: {str(e)}")
    finally:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()