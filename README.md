# 동심을 찾아서: 아이들을 위한 창의적 디지털 놀이터  

"동심을 찾아서"는 어린아이들의 상상력을 디지털 세상에서 실현시키는 창의적인 플랫폼입니다.  
이 프로젝트는 아이들이 직접 그린 스케치를 다음과 같은 과정을 통해 생동감 넘치는 애니메이션으로 완성합니다:  

Step1. **스케치의 엣지 추출 및 채색 이미지 생성**

Step2. **다양한 배경 추가**  

Step3. **캐릭터에 동적인 모션 부여** 

아이들은 자신만의 그림이 살아 움직이는 모습을 보며 상상력과 창의력을 마음껏 발휘할 수 있습니다.  

[프로젝트 발표자료](https://drive.google.com/file/d/1AlGLFn5aqtn0KXdo1vzc8_NqJGljM6wa/view?usp=sharing)

---
**2024.11.21 - 2024.12.27**

<table>
  <tbody>
    <tr>
      <td align="center">
        <a href="https://github.com/navi0728">
          <img src="https://github.com/navi0728.png" width="100px;" alt=""/>
          <br /><sub><b>MinJu Lee</b></sub>
        </a>
        <br />
      </td>
      <td align="center">
        <a href="https://github.com/winnercalvin">
          <img src="https://github.com/winnercalvin.png" width="100px;" alt=""/>
          <br /><sub><b>SeungHo Park</b></sub>
        </a>
        <br />
      </td>
      <td align="center">
        <a href="https://github.com/rwambangho">
          <img src="https://github.com/rwambangho.png" width="100px;" alt=""/>
          <br /><sub><b>ByeongHo Yoon</b></sub>
        </a>
        <br />
      </td>
    </tr>
  </tbody>
</table>

---

## 배경  
이 프로젝트는 여수 **아르떼 뮤지엄**에서의 경험에서 영감을 받았습니다.  
어린이가 그린 그림을 채색한 뒤, 이를 컴퓨터에 입력하면 3D로 변환되어 가상 수족관 속에서 움직이는 모습을 볼 수 있는 프로그램을 통해 깊은 감명을 받았습니다.  
이를 바탕으로 더 다양한 표현 방식과 주제를 통해 아이들에게 창의력을 자극하는 디지털 플랫폼을 개발하고자 합니다.  

---

## 프로젝트 단계  

### **Step 1: Sketch2Image**  
- **Input**: 스케치 이미지,promt(option) 
- **Output**: 채색된 이미지 
- **사용 모델 inference**: [ControlNet](https://github.com/lllyasviel/ControlNet-v1-1-nightly)  

### **Step 2: Image2Background**  
- **Input**: 채색된 이미지,prompt(option)  
- **Output**: 가상 배경이 생성된 이미지  
- **사용 모델 inference**: [Photo Background Generation](https://github.com/yahoo/photo-background-generation?tab=readme-ov-file)  

### **Step 3: Background2Movement**  
- **Input**: 가상 배경이 생성된 이미지,prompt(option) 
- **Output**: 2초 애니메이션 (GIF)  
- **사용 모델 inference**: [Animate Anything](https://github.com/alibaba/animate-anything)  

---

## 데이터셋  
- [Sketch Dataset](https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/)  

---

## 사용 기술
- **Generative Model**
- **Stable Diffusion with ControlNet**
- stabel diffusion: **Step1) Animate Anything v4.5, Step2) Stable Inpainting 2.5**
- condition: **Canny Edge Detection, Salient Object Detected Instance**
---

## 결과 이미지  
### Sketch Image
<img src=https://github.com/navi0728/Sketch2Movement/blob/main/src/Sketch_Image.png width="200" height="200"/>

### Step 1: Sketch2Image  
<img src=https://github.com/navi0728/Sketch2Movement/blob/main/src/Step1_output.png width="200" height="200"/>

### Step 2: Image2Background  
<img src=https://github.com/navi0728/Sketch2Movement/blob/main/src/Step2_output.png width="200" height="200"/>

### Step 3: Background2Movement  
<img src=https://github.com/navi0728/Sketch2Movement/blob/main/src/Step3_output.gif width="200" height="200"/>


