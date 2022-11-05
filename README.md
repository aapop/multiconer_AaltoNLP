# AaltoNLP system at SemEval 2022 Task 11: MultiCoNER

This repository presents solution of AaltoNLP (Aapo Pietiläinen & Shaoxiong Ji) for SemEval 2022 Task 11: MultiCoNER. The task is about complex Named Entity Recognition in multilingual context.

The system is built on top of the baseline provided by the organizers https://github.com/amzn/multiconer-baseline.

This repository contains implementations for the two systems we present in our system description paper https://aclanthology.org/2022.semeval-1.202/.


Our system proposes to ensemble strategies using multilingual transformer models:
1. Naive ensemble
![](figures/naive_ensemble.png?raw=true)
2. End-to-end ensemble
![](figures/e2e_ensemble.png?raw=true)




### Setting up the code environment

```
$ pip install -r requirements.txt
```

### Training the model

```
$ python train_model.py --train data/EN-English/en_train.conll --dev data/EN-English/en_dev.conll --out_dir --model_name koala/xlm-large-en --gpus 1 --epochs 3 --encoder_model koala/xlm-roberta-large-en --batch_size 32 --lr 0.0001
```
### Evaluating the model

```
$ python evaluate.py --test data/EN-English/en_dev.conll --out_dir results --gpus 1 --encoder_model koala/xlm-roberta-large-en --model INSERT_MODEL_HERE
```

### Predicting tokens

```
$ python predict_tags.py --test data/EN-English/en_dev.conll --out_dir predictions --encoder_model koala/xlm-roberta-large-en --model INSERT_MODEL_HERE
```

### Predict with naive ensemble

When you have trained sufficient amount models, add them into the list in **predict_ensemble_sm.py** and run the file with test file arguments.

### E2E ensemble models

Run the steps similarly but use ensemble model files

```
$ python train_ensemble.py ...
$ python evaluate_ensemble.py ...
$ python predict_ensemble_e2e.py ...
```
