# Temporal Topic Modelling with Pachinko Allocation

This repository contains code for the modelling part of the ["A Sustainable West? Analyzing Clusters of Public Opinion in Sustainability Western Discourses in a Collection of Multilingual Newspapers (1999-2018)"](https://doi.org/10.5281/zenodo.7742461) paper.

## Content
It includes code for the *Topic Modelling*, *Topic Clustering* and *Topic Evolution*.

1. `pachinko_model.ipynb` uses the Pachinko Allocation model to find topics in a set of documents (it also contains information on the finetuning of parameters and sentiment analysis,
2. `topic_similarity.ipynb` analyzes the relationships between the topics of different corpora (that were identified by PA model, you can apply same pipeline to other Topic Modelling Models) - topic similarity, topic clustering, topic evolution and sentiment analysis. 
                   
                  

## How to cite 
**Zenodo Preprint**
```bibtex
@article{fernandez2023sustainable,
  author       = {Elena Fernández Fernández and Germans Savcisens},
  title        = {{"A Sustainable West? Analyzing Clusters of Public Opinion in Sustainability Western Discourses in a Collection of Multilingual Newspapers (1999-2018)"}},
  month        = mar,
  year         = 2023,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.7742461},
  url          = {https://doi.org/10.5281/zenodo.7742461}
}
```
