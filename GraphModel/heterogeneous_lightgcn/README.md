# GNN을 이용한 DKT - author: [권준혁](https://github.com/tree-jhk)

## LightGCN baseline
- NGCF에서 Message Passing 과정을 더욱 간단히 해서 경량화를 진행한 LightGCN 모델을 사용했습니다.
- Pytorch-geometric 라이브러리의 LightGCN을 활용했습니다.
## Edge dropout을 적용한 LightGCN
![image](https://user-images.githubusercontent.com/97151660/208237990-ad1f04ff-5311-4737-adaa-2149075adaf3.png)
- Edge dropout: 원본 graph의 node를 삭제하지 않고, edge들을 랜덤하게 삭제해서 매번 새로운 graph로 학습하게 하는 효과를 줍니다.
- 이를 통해 동일한 graph가 다양한 형태의 graph인 것처럼 매번 학습되기 때문에 robust해지고 data augmentation 효과도 확보합니다.
## Quadripartite Heterogeneous Graph Propagation for DKT (QHGP) (제작한 모델)
![image](https://user-images.githubusercontent.com/97151660/208238060-6aea02d9-f347-4f91-9a67-f7682313f0c1.png)
  - figure by **[권준혁](https://github.com/tree-jhk)**
- 학생(userID) - 문항(problemID) 간의 관계뿐만 아니라, 학생(userID) - 개념(KnowledgeTagID) 간의 관계와 학생(userID) - 시험지(testID) 간의 관계 그래프도 분명히 존재하기에 이 세 가지 bipartite graph를 연결해서 heterogeneous graph를 구성하고 싶었습니다.
- Pytorch-geometric 라이브러리의 LightGCN을 직접 불러와서 수정하는 작업을 진행했습니다.
## Quadripartite Heterogeneous Graph Propagation with features for DKT
- 기존 LightGCN의 협업 필터링 효과에 여러 feature engineering 정보(user별 정답률 평균, tag별 정답률 분산 등과 같은 연속형 통계량이나 범주형 통계량들)를 추가해서 모델이 더 많은 힌트를 받고 성능이 오를 것으로 기대했습니다.
- 그러나 overfitting 되는 경험을 했습니다.
## AUROC Compare
![image](https://user-images.githubusercontent.com/97151660/208237934-7c2bb946-4b1e-4ee5-9c95-32cc428be07b.png)


# 동작환경

- cuda verion ==11.3
- torch==1.10.0
- torch-scatter==2.0.9
- torch-sparse==0.6.13
- torch-geometric==2.0.4

라이브러리 설치

```
pip uninstall torch, torch-sparse, torch-scatter, torch-geometric
pip3 install torch==1.10.0
pip install torch-scatter==2.0.9
pip install torch-sparse==0.6.13
pip install torch-geometric==2.0.4

라이브러리 설치가 잘 안된다면 다음을 시도해보세요: by 권준혁
print(torch.version.cuda), print(torch.__version__) 확인해서 버전에 맞는 cuda, torch를 설치해주세요.

pip install torch==1.10.0
pip install torch-sparse==0.6.13 -f https://pytorch-geometric.com/whl/torch-1.10.0+cu102.html
pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.10.0+cu102.html
pip install torch-geometric==2.0.4
```


# 파일

- install.sh : 관련 library 설치 스크립트 파일
- config.py : 설정 파일
- lightgcn/datasets.py : 데이터 로드 및 전처리 함수 정의
- lightgcn/model.py : 모델을 정의하고 manipulation 하는 build, train, inference관련 코어 로직 정의
- lightgcn/utils.py : 부가 기능 함수 정의
- train.py : 시나리오에 따라 데이터를 불러 모델을 학습하는 스크립트
- inference.py : 시나리오에 따라 학습된 모델을 불러 테스트 데이터의 추론값을 계산하는 스크립트
- evaluation.py : 저장된 추론값을 평가하는 스크립트


# 사용 시나리오

- install.sh 실행 : 라이브러리 설치(기존 라이브러리 제거 후 설치함)
- config.py 수정 : 데이터 파일/출력 파일 경로 설정 등
- train.py 실행 : 데이터 학습 수행 및 모델 저장
- inference.py 실행 : 저장된 모델 로드 및 테스트 데이터 추론 수행
