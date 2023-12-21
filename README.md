### Project Title

현악기 음원 미디 음악 전사 프로젝트 

### Overview

- 기간  |  2021. 07 ~ 2021. 07
- 담당 파트 |  개인프로젝트
- 플랫폼 | Python, Tensorflow, Colab notebook

### Background 

1. 피아노 연주 전사 모델인 Magenta Onsets and Frames 모델을 활용하여 현악기 등의 다양한 악기에 적용가능 여부를 검토하고자 함
2. 피아노와는 다른 파형을 지닌 현악기에 맞는 전처리와 모델링 가능성을 검토하고자 함

### Goal

1. Onsets and Frame 논문에 따른 모델 구현
2. 현악기 연주의 녹음 파일(WAV)을 입력 받아 MIDI 파일로 출력

### Dataset

1. Maestro
  - 100G 분량의 피아노 Data
  - WAV 파일과 이에 대응하는 MIDI Data Set
  - 연도별 파일 중 2014, 2015, 2017 Data를 사용(RAM 사양 문제로 많은 데이터를 처리할 수 없는 한계가 있음)

2. MusicNet
  - 10G 분량의 클래식 연주 Data
  - WAV 파일과 이에 대응하는 MIDI Data Set
  - 현악기 Data만 추출하여 사용(122 WAV/MIDI)

### Theories

1.  Maestro Data(2014, 2017)와 MusicNet Data를 Training한 결과는 Maestro Data(2014, 2015, 2017)만으로 Training한 결과보다 Transcription F1 Score가 높을 것이다.
2. Maestro Data를 Training한 모델에 MusicNet Data를 Transfer-Learning한 결과는 Base Model보다 Transcription F1 Score가 높을 것이다.

### Sound Wave

1. 소리란 공기나 물 같은 매질의 진동을 통해 전달되는 파동을 의미
2. Wave Form
  - 진폭(amplitude) : 진동 중심에서 최대 변위의 크기(소리의 세기)
  - 파장(wavelength) : 같은 변위(위상)을 가진 서로 이웃한 두 점 사이의 거리
  - 주기(period) : 매질의 한 점이 1회 진동하는데 걸린 시간
  - 진동수(frequency) : 파동 전파시 매질의 한 점이 1초 동안 진동한 횟수(소리의 높낮이)
    
<p align="center">
  <img src="https://github.com/mugan1/music_transcription/assets/71809159/1354e688-eca5-41ba-9222-e246817d9751" alt="text" width="number" />
</p>

### 푸리에 변환(Fourier transform)

<p align="center">
  <img src="https://github.com/mugan1/music_transcription/assets/71809159/842216db-9eef-4b16-a1c3-473b89b5ef24" alt="text" width="number" />
</p>

1. 음원 데이터는 녹음 장치에 전달되는 순간적인 음압(sound pressure)만을 측정해서 만들어짐
2. 푸리에 변환은 임의의 입력 신호를 다양한 주파수를 갖는 주기함수들의 합으로 분해하여 표현하는 것(sin, cos함수)
3.  waveform의 시간 영역(time domain)에서 주파수 영역(frequency domain)의 함수로 변환

### Mel-Spectrogram

<p align="center">
  <img src="https://github.com/mugan1/music_transcription/assets/71809159/404b20b8-537a-4483-aa94-45db0b4cce36" alt="text" width="number" />
</p>

<p align="center">
  <img src="https://github.com/mugan1/music_transcription/assets/71809159/c9fccbe1-30bc-4028-8bb3-7c3bd7c73234" alt="text" width="number" />
</p>

1. STFT(Short Time Fourier Transform)
  - 푸리에 변환은 시간의 흐름에 따라 신호의 주파수가 변했을 경우 어느 시간대에 주파수가 어떻게 변했는지를 알 수가 없음
  - 주파수 성분이 시간의 흐름에 따라 어떻게 변하는지 분석하는 방법
2. Mel-Spectrogram
  - STFT에서 사람이 음성 신호를 인식하는 기준(낮은 주파수를 높은 주파수보다 더 예민하게 받아들임)으로 Scaling한 결과
  - Onsets and Frames 모델 내의 CNN 모델의 Data로 활용됨

### MIDI(Musical Instrument Digital Interface)

1. 각기 다른 악기를 공통된 전자 언어를 사용하여 서로 소통할 수 있도록 설계된 컴퓨터 네트워크
2. 주요 Message
  - Note_On / Note_Off : 소리의 시작/끝을 의미
  - Note_Value : 음계 또는 Key를 의미
  - Velocity : "타건강도“로서 건반을 내리치는 힘 또는 속도를 의미

### 피아노와 현악기의 특성

<p align="center">
  <img src="https://github.com/mugan1/music_transcription/assets/71809159/1515f4bf-e140-4943-b590-7fc2eabf3d4e" alt="text" width="number" />
</p>

1. 피아노는 건반을 내리치는 방법으로 소리를 내며 현악기는 활로 현을 진동시키는 방식
2. 피아노는 건반을 내리칠 때 가장 진폭이 크게 나타나며, 타건강도를 내는 Velocity 또한 피아노 음원을 분석하는 데에 영향을 미침
3. 같은 주파수(음계)를 연주하더라도 악기의 배음 구조의 차이로 인해 다른 음색, 다른 파형을 지니게 됨

### Onsets and Frames 모델 Concept

<p align="center">
  <img src="https://github.com/mugan1/music_transcription/assets/71809159/8ca0baab-f4eb-4adb-b425-616df7edb151" alt="text" width="number" />
</p>

1. Note의 시작을 감지하는 Onset Detector와 활성화 된 모든 Note를 감지하는 Frame Detector가 존재 -> 이전 하나의 Stack만 존재한 모델 보다 성능이 향상 됨
2. 2차원 Mel-Spectrogram을 입력 받는CNN 모델과 BiLSTM(양방향 LSTM) 모델로 구성
3. Onset Detector는 정확히 Note의 시작을 감지했을 때 메모를 시작할 수 있도록 제한을 주는 역할 
4. 모델의 가중치는 Note의 초기 Frame에 가중치를 높게 줌
   
### Modeling

<p align="center">
  <img src="https://github.com/mugan1/music_transcription/assets/71809159/f2f49777-48bb-4957-80c0-2f9275f8adde" alt="text" width="number" />
</p>

1. Onset Network
2. Offset Network : Onset과 상응하는 Network
3. Frame Network : Onset과 Offset Network를 결합하여 만듦
4. Velocity Network : 별도의 Network로 학습
   
### Metrics

<p align="center">
  <img src="https://github.com/mugan1/music_transcription/assets/71809159/10f05f4d-b61b-4d8c-a2e6-a9517f92b2a0" alt="text" width="number" />
</p>

1. 모델의 정확도는 accuracy의 경우 Dixon의 evaluation Metrics 사용
2. Testing을 위한 F1 Score는 MIR-EVAL(Music Information Retrieval) Library 사용

### Training and Tesing Plan

<p align="center">
  <img src="https://github.com/mugan1/music_transcription/assets/71809159/a02bdbff-8072-4b1b-91cf-0d91c7ba41c0" alt="text" width="number" />
</p>

1. Base Model : Magenta에서 구현한 Checkpoint를 그대로 적용시킨 모델
2. Test2/Test3 : Customized Model
3. Test4 : Base Model에서 MusicNet 현악기 데이터를 Transfer-Learning 시킨 모델

### Result

<p align="center">
  <img src="https://github.com/mugan1/music_transcription/assets/71809159/7b79b58f-47a6-4240-96b4-e7520bcb22ce" alt="text" width="number" />
</p>

<p align="center">
  <img src="https://github.com/mugan1/music_transcription/assets/71809159/eeaea5a9-58fb-4521-a88c-556fdd8ff36c" alt="text" width="number" />
</p>


### Conclusion

1. 총평

  - Customized 모델은 Base 모델의 성능에 미치지 못했으나, Musicnet Data를 함께 훈련시킨 결과가 Maestro Data만을 훈련시킨 F1 Score보다 높게 나타남
  - Transfer Learning 모델은 Base 모델의 F1 Score보다 높은 수치를 달성

2. 소감 및 기대효과

  - MusicNet 현악기 연주 데이터의 양이 많이 부족하며, 현악기 특성에 맞는 Feature Extraction(데이터 전처리)과 모델링이 필요성을 느낌
  - Test 모델들은 대부분 음정과 박자 등의 기본요소를 Transcription 할 수 있던것으로 보아 보완사항들을 해결한다면 완성도 높은 현악기 연주 음악 MIDI 파일을 전사할 수 있을 것이라 예상함
    
### References

1. Google Magenta Project : [Link](https://github.com/magenta/magenta/tree/master/magenta/models/onsets_frames_transcription)
2. Magenta Transcription Github : [Link](https://github.com/BShakhovsky/PolyphonicPianoTranscription)

