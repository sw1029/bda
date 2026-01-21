https://dacon.io/competitions/official/236664/overview/description

대회 실험용 레포

변수 명명법은 스네이크 형식으로 통일한다
낙타 표기법으로 쌍봉낙타마냥 변수가 iAmSoSmart 이러다간 미간에 지건을 꽂아버릴것

gpu가 빠방한 경우 아래의 명령어로 torch까지 설치(선택사항. 무거우니까 왠만하면 보류 추천)

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118




개선예정사항

1. model 모듈 팩토리 패턴 적용
2. model 모듈에서 학습 시 경로 인자 제공하는 경우 자동 저장 기능 추가
3. 데이터 전처리, 시각화 관련 기능 추가
4. 여러 시드값 결과 종합 앙상블 기능 추가.