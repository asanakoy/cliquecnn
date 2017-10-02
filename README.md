# CliqueCNN: Deep Unsupervised Exemplar Learning (NIPS 2016)

Based on our NIPS 2016 Paper: **"CliqueCNN: Deep Unsupervised Exemplar Learning"** by Miguel A. Bautista* , Artsiom Sanakoyeu* , Ekaterina Sutter, Björn Ommer.

https://asanakoy.github.io/cliquecnn/

---

* The paper can be downloaded from https://arxiv.org/abs/1608.08792
* Labels that we gathered for Olympic Sports can be found in [olympic_sports_retrieval/data](olympic_sports_retrieval/data)
* All our pretrained models for Olympic Sports dataset can be downloaded from [here](https://hcicloud.iwr.uni-heidelberg.de/index.php/s/kRp6b454Dd0wnts)
* Caffe's deploy file: [olympic_sports_retrieval/models/deploy.prototxt](olympic_sports_retrieval/models/deploy.prototxt)  
* Evaluation script for Olympic Sports: [olympic_sports_retrieval/calculate_roc_auc.py](olympic_sports_retrieval/calculate_roc_auc.py)
* Baseline HOG-LDA similarity matrices for Olympic Sports:
[similarities_hog_lda.tar.gz](http://compvis10.iwr.uni-heidelberg.de/share/cliquecnn/similarities_hog_lda.tar.gz) (11.5 Gb)

If you find this code or data useful for your research, please cite
```
@inproceedings{cliquecnn2016,
  title={CliqueCNN: Deep Unsupervised Exemplar Learning},
  author={Bautista, Miguel A and Sanakoyeu, Artsiom and Tikhoncheva, Ekaterina and Ommer, Bj{\"o}rn},
  booktitle={Proceedings of the Conference on Advances in Neural Information Processing Systems (NIPS)},
  pages={3846--3854},
  year={2016}
}
```
