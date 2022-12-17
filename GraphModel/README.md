# GNN을 이용한 DKT - author: [권준혁](https://github.com/tree-jhk)

## LightGCN baseline
- NGCF에서 Message Passing 과정을 더욱 간단히 해서 경량화를 진행한 LightGCN 모델을 사용했습니다.
- Pytorch-geometric 라이브러리의 LightGCN을 활용했습니다.
## Edge dropout을 적용한 LightGCN
- ![image](https://user-images.githubusercontent.com/97151660/208237990-ad1f04ff-5311-4737-adaa-2149075adaf3.png)
- Edge dropout: 원본 graph의 node를 삭제하지 않고, edge들을 랜덤하게 삭제해서 매번 새로운 graph로 학습하게 하는 효과를 줍니다.
- 이를 통해 동일한 graph가 다양한 형태의 graph인 것처럼 매번 학습되기 때문에 robust해지고 data augmentation 효과도 확보합니다.
## Quadripartite Heterogeneous Graph Propagation for DKT (QHGP) (사용한 모델)
- ![image](https://user-images.githubusercontent.com/97151660/208238060-6aea02d9-f347-4f91-9a67-f7682313f0c1.png)
  - figure by **[권준혁](https://github.com/tree-jhk)**
- 학생(userID) - 문항(problemID) 간의 관계뿐만 아니라, 학생(userID) - 개념(KnowledgeTagID) 간의 관계와 학생(userID) - 시험지(testID) 간의 관계 그래프도 분명히 존재하기에 이 세 가지 bipartite graph를 연결해서 heterogeneous graph를 구성하고 싶었습니다.
- Pytorch-geometric 라이브러리의 LightGCN을 직접 불러와서 수정하는 작업을 진행했습니다.
## Quadripartite Heterogeneous Graph Propagation with features for DKT
- 기존 LightGCN의 협업 필터링 효과에 여러 feature engineering 정보(user별 정답률 평균, tag별 정답률 분산 등과 같은 연속형 통계량이나 범주형 통계량들)를 추가해서 모델이 더 많은 힌트를 받고 성능이 오를 것으로 기대했습니다.
- 그러나 overfitting 되는 경험을 했습니다.
## AUROC Compare
![image](https://user-images.githubusercontent.com/97151660/208237934-7c2bb946-4b1e-4ee5-9c95-32cc428be07b.png)
