TODO checklist입니다^^

각 요구사항별로 세부 설명도 포함해서 적어둠
양식은 MoSCoW 따라하긴 했는데, 그냥 보고 느낌껏 구현해도 무방.
구현 끝내면 todo 문서 [ ] 안에 x 채워넣기(일일퀘스트 느낌)

**중요도 우선순위**
- MUST: 필수사항
- SHOULD: 필수는 아니지만 중요함
- COULD: 하면 좋음
- WONT: 이거까진 할 필요 없음. 혹은 지양
  
  

[ ] k-fold 구현
- MUST
  : config에 작성된 변수를 읽어와서 그 횟수만큼 k-fold 수행(from sklearn.model_selection import StratifiedKFold 사용 권장). 
- SHOULD
  : 모델 안에 timestamp에 따라 생성되는 dir안의 저장에 대하여 k-fold의 fold마다의 결과물을 이름 구분하여 저장
- COULD
  : 가능하면 k-fold 후 결과 집계/종합 기능 구현(귀찮다 싶으면 utils.py 내에 구현하고 붙여도 무방). 왠만하면 train.py에서의 호출 양식은 최대한 유지. train_test_split 호출 부분 근처에서 k-fold 구현한것 호출 외에는 유지되면 좋겠음.
- WONT
  : 딱히 없음. 화이팅.
  



[ ] 다중 seed 지정
- MUST
  : k-fold와는 별개로 train에서 set_seed 호출 시 cfg에서 시드가 여러 개 설정되어 있는 경우, 혹은 multi seed 관련 플래그가 on으로 설정되어 있는 경우 동일 timestamp 기준 dir 내에 seed별로 구분되어 저장되도록 구현.
- SHOULD
  : seed별 결과물 후처리, 집계 util 구현
- COULD
  : seed별 집계에서 여러 집계 방식, seed별 결과 분포 및 bias가 어느 정도로 존재하는지 분석.
- WONT
  : 왠만하면 지금 train.py 내의 호출 형태를 최대한 유지하면서 진행하면 좋겠음. 



[ ] 사용 방법론별 저장 파일 이름 변경 구현
- MUST
  : 결과 저장 dir에서 이름만으로 사용 방식 식별 가능하도록 저장 이름이 구분되도록 이름 저장 기능 구현. 이미 catboost 에서 구현이 되어있는 저장분기(save_dir = save_dir/ "catboost" / f"{self.timestamp}") 참고하여 추가 구분 기능 구현
- SHOULD
  : 이름 명명 방식 별도 정의
- COULD
  : 명명된 이름 기반 결과 조회, 집계 함수 구현
- WONT
  : 이름이 지나치게 길어지는건 가급적 지양. 손으로 입력할 때 힘들다...




[ ] best threshold 분석 util 구현
- MUST
  : 모델의 출력이 regression 기반 실수인 경우 valid data 기준 best threshold 값을 후처리로 확인하는 기능 구현
- SHOULD
  : seed별 best threshold
- COULD
  : threshold 근방에 존재하는 값들의 로그 별도로 확인 기능, 사용 방법론별 best threshold 값 집계 및 시각화 기능
- WONT
  : 딱히 없음.




[ ] MLP 구현
- MUST
- SHOULD
- COULD
- WONT
  



[ ] 증강 구현
- MUST
- SHOULD
- COULD
- WONT