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
    
내부 소스의  
InputLayer (날개, 꼬리, 엔진 등) 와 ConjunctionLayer (규칙1, 규칙2 등) 등과 질문을 바꾸고  
학습데이터를 알맞게 넣는 다면, 다른 결과를 이끌어 내는 것도 가능할 듯

