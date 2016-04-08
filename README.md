# Artificial-Neural-Net-Simple-Example
파이썬을 이용한 간단한 인공신경망 예제  
    
처음 가중치는 랜덤으로 초기화 시키고, 학습 데이터를 이용해 가중치를 조절한다.  

yes -> 있을 경우  
no -> 없을 경우  
dunno -> 모를 경우  

Ex)  

글라이더는 엔진과 깃털이 없습니다  

    날개가 있나요? yes/no/dunno : yes
    부리가 있나요? yes/no/dunno : no
    꼬리가 있나요? yes/no/dunno : dunno
    깃털가 있나요? yes/no/dunno : no
    엔진가 있나요? yes/no/dunno : no
    결과 : 글라이더

비행기는 엔진이 있으며,  

    꼬리가 있나요? yes/no/dunno : yes
    부리가 있나요? yes/no/dunno : no
    날개가 있나요? yes/no/dunno : yes
    깃털가 있나요? yes/no/dunno : no
    엔진가 있나요? yes/no/dunno : yes
    결과 : 비행기
    
새는 부리, 꼬리, 깃털 등이 있으므로  

    날개가 있나요? yes/no/dunno : yes
    꼬리가 있나요? yes/no/dunno : yes
    엔진가 있나요? yes/no/dunno : no
    부리가 있나요? yes/no/dunno : yes
    깃털가 있나요? yes/no/dunno : yes
    결과 : 새

([학습데이터](https://github.com/Kcrong/Artificial-Neural-Net-Simple-Example/blob/master/train_data.txt)의 경우 임의로 적어놓은 값입니다.)

자연어 기능

결과 예시 1)

    엔진소리가 들리지 않고, 꼬리가 보이지 않는다. 부리가 없다. 또한 깃털도 안보인다
    {'깃털': -1, '꼬리': -1, '엔진': -1, '날개': 0, '부리': -1}

결과 예시 2)

    엔진과 날개가 있으며  꼬리는 모르겠다. 부리가 안보인다. 아니다, 꼬리가 있다.
    {'깃털': 0, '꼬리': 1, '엔진': 1, '날개': 1, '부리': -1}

결과 예시 3)

    글라이더 같은 데, 잘 알 수 없고 꼬리가 없다. 엔진 소리가 안 들리고 날개가 있다
    {'깃털': 0, '꼬리': -1, '엔진': -1, '날개': 1, '부리': 0}

실행 및 결과 예시)

    특징을 입력해주세요 : 깃털이 있고, 부리도 있다. 날개와 꼬리가 있고 엔진 소리는 들리지 않는다.
    {'깃털': -1, '꼬리': -1, '엔진': -1, '날개': -1, '부리': -1}
    새