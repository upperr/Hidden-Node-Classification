# Hidden-Node-Classification
This is the source code of our work "Boosting Hidden Graph Node Classification for Large Social Networks".

## Summary
In real-world social networks, there usually exist a large number of hidden nodes, such as the newly appeard users in evolving networks and the unobservable graphs in cross-graph networks. Aiming at the classification of hidden nodes, we propose a deep generative model based on the variational auto-encoders (VAEs) for graph-structured data. The model employs a graph neural network (GNN) as encoder to aggregate multi-hop neighborhood information of each node. Meanwhile, in order to make better use of the structural information for graphs with sparse node features, the proposed model adopts a novel degree-gated attention mechanism to assigns various weights to each neighbor according to their influence. In addition, our model can be trained by minibatches and thus is applicable for large-scale networks. The experimental results based on several real-world social network datasets verify the effectiveness of our model for hidden node classification.

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
