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

---

## 프로젝트 회고
본 프로젝트 과정에서는 세가지 문제점을 가지고 있었습니다. 

첫번째 문제점은 스케치 이미지를 3D이미지로 생성하는 단계에서 **스케치 이미지의 정보를 크게 반영하지 못하는 것** 이었습니다.
이를 해결하기 위해 Stable diffusion의 입력조건을 넣는 부분에서 Lineart Anime모델을 사용하여 입력된 이미지의 **선화(lineart)를 추출**함으로써 이미지의 구조와 형태를 파악할 수 있어 해당 구조를 유지하면서도
의도하고자 한 애니메이션 스타일의 색상과 디테일적인 부분을 추가해주었기에 최종적으로 원하는 결과를 만들 수 있었습니다.

<img width="775" height="503" alt="image" src="https://github.com/user-attachments/assets/f1dd72a6-01a7-45b2-b605-6f40393a8bea" />


두번째 문제점은 배경을 생성하는 단계로 넘어갈때 3D로 생성하는 단계에서 **선화에 맞게 채색이 안된 경우 잘못 채색된 이미지를 기준으로 마스크를 생성**하여 그것이 배경 생성에 반영된다는 문제점이 있었습니다. 
이를 해결하기 위해 스케치 이미지를 기준으로 마스크를 생성하였지만 스케치의 선이 떨어져 있는 경우는 적용이 불가해 OpenCV의 Dilation과 Erosion을 이용하여 **선의 굵기를 조절하는 후처리 작업**을 통해 해당 문제를 해결할 수 있었습니다.

<img width="515" height="341" alt="image" src="https://github.com/user-attachments/assets/19528a03-2aff-4774-9dc1-d2521d2f6328" />

세번째 문제점은 한정적인 자원으로 인한 한계입니다. 직접 그려서 테스트했을시에도 우수한 성능을 보였기에 수집하고 정제한 데이터셋으로 전이학습 및 파인튜닝을 진행하려 했으나 OOM발생으로 인해 **프롬프트 엔지니어링을 통해** 성능을 개선하기로 결정했습니다. 
프롬프트의 의존성이 높아 입력 이미지의 정보를 최대한 반영시키기 위해 하이퍼파라미터 값을 조정하였고 Controlnet의 논문을 분석하여 여러가지 프롬프트를 실험한 결과 **명사를 나열하거나 동작을 추가해주었을 때** 좋은 결과가 나오는 것을 확인할 수 있었습니다.

<img width="501" height="418" alt="image" src="https://github.com/user-attachments/assets/268154de-e818-4656-ae5a-582699b896c1" />

또한 배경 생성 단계에서는 condition_scale파라미터를 1.0으로 조정했을 경우 마스킹된 객체의 형태와 색을 그대로 유지할 수 있었고 마스킹된 객체와 배경 설명을 디테일하게 한 프롬프트를 입력했을 때 가장 깔끔하게 나오는 것을 
확인할 수 있었습니다.

<img width="918" height="362" alt="image" src="https://github.com/user-attachments/assets/7bb43dea-be13-4c47-8d07-d25f269731ce" />

