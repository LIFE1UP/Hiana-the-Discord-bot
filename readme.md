# 하이아나는 딥러닝 디스코드 봇

## 활성화 시키기
1. bot.py를 열고 Discord API key 부분에 지정받은 **token**을 넣어줍니다. <br />
2. 터미널을 열고 $python bot.py를 타이핑합니다. <br />
3. 하이아나가 동작됩니다.

## 훈련시키기
1. 'txbs/textbook.json'에서 데이터를 이해하고 적절하게 바꾸면; 새로운 문맥이나 대사를 추가할 수 있습니다. <br />
2. 터미널을 열고 $python train.py를 타이핑합니다. <br />
3. 하이아나가 학습을 합니다.

## 기능을 추가하기
response.py를 편집해서 기능을 추가할 수 있습니다. 러닝타임에서 모든 기능은 사전으로 접근되어지며; 그 기능의 함수는 스트링을 반환해야만 합니다. <br />
> 스포티파이 차트, 재생목록을 기준으로 음악을 추천해주는 모듈이 기본으로 들어있습니다. 정해진 CSV를 추천하는 기능과 스포티파이의 플레이 리스트를 동적으로 받아서 추천하는 기능이 있습니다.
> 활성화 함수를 바꾸고 싶으면 model.py에서 편집합니다. 기본 세팅은 시그모이드를 사용합니다.

# Hiana the Deep Learning Discord Bot

## Activate the Bot
1. edit bot.py to add **Discord API Key**(Token)
2. $python bot.py to activate Hiana

## Teach the Bot
1. edit 'txbs/textbook.json' after understanding its structures; you can add new context or line to response
2. $python train.py to make the bot learn them

## Add new Function
from response.py, you can add or delete function. Every function may pass through **Dic**; and should return string type
> there is a music recommendation module for default; one of it is uses URL, the other uses CSV
