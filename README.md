# Hidden-Node-Classification
This is the source code of our work "[Boosting Hidden Graph Node Classification for Large Social Networks](https://ieeexplore.ieee.org/abstract/document/9624788)".

## Summary
Identifying hidden nodes in social networks is a critical issue in security-related applications. In contrast to the conventional node classification on graphs with all nodes being observable, it is more challenging to classify the hidden nodes that are unobservable during the training process, also known as the "inductive learning" in previous research. Existing approaches for inductive node classification mainly adopt graph neural network models to learn node representations. Although these methods are advantageous to modeling the topology of graph-structured data, they rely heavily on node features which may vary significantly in different specific application scenarios. In addition, the inherently changeable graph structure induced by hidden nodes may cause the over-fitting problem. To address the above issues and boost the performances of hidden node classification, we propose a deep generative model based on variational auto-encoders. Specifically, we design a novel graph neural network to aggregate the multi-hop neighbor information of each node. Meanwhile, to better utilize the graph structure information as a supplement to node features, we consider the heterogeneous node influences and introduce a gated attention mechanism using node degrees. Moreover, our proposed model can be trained by minibatches and thus is applicable to large social networks. We conduct experiments on four real-world datasets, and verify the effectiveness of our method for hidden graph node classification.

## Requirements

* python>=3.7
* networkx==1.11
* numpy==1.19.5
* sklearn==0.0
* tensorflow==2.2.0

## Example
python train.py --dataset reddit --epochs 100 --encoder 128_64 --decoder 50

## Datasets
All datasets used in our paper are available for download:

* Reddit
* Elliptic
* Flickr
* Deezer
* ... (more to be added)

They are available at [BaiduYun link (code: grcu)](https://pan.baidu.com/s/1pBV6svzp-uQuSuyx5F2joQ). Unzip the package at the root directory.

If you make use of this code in your work, please cite the following paper:

     @inproceedings{yang2021boosting,
                    title = {Boosting hidden graph node classification for large social networks},
                    author = {Yang, Hanxuan, Kong, Qingchao, Mao, Wenji and Wang, Lei},
                    booktitle = {IEEE International Conference on Intelligence and Security Informatics},
                    pages = {1--6},
                    year = {2021}
	   }
