# CANDELA
Code for the publication "Cancer Drug Sensitivity Estimation using Modular Deep Graph Neural Networks"

`run_experiments.sh` contains the driver code for running the both pretraining tasks and the proposed models:
- `data_preprocessing.py` contains the code for the data preprocessing for the pretraining and downstream tasks. The data (and pretrained models) can be found in https://zenodo.org/record/8020946
- `pretrain_features.py` stores the weights of the model trained to predict molecular features
- `pretrain_toxicity.py` stores the weights of the model trained to predict molecular toxicity
- `train_gnns.py` trains the models using early stopping, with different configurations specified as command line arguments
- `predict_gnns.py` takes the models train on the previous step and generates the predictions
- `python3 integrated_gradients.py` stores the feature importances obtained from running integrated gradients
