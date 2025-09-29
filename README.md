# alc-paper

## Datasets

The datasets used in the paper can be found in directory `original_data_parquet`. The datasets in `weak_data_parquet` and `strong_data_parquet` are generated using `prep_anon.py`. 


### Code

Directory `dependence` contains the code and results for generating the plot showing the effect of dependent rows in the data.

Directory `compare` contains the code for running the attacks using both our approach and a prior approach similar to Giomi et al.'s Anonymeter.

Directory `prc_alc`contains the code generating the plots showing the behavior of the PRC and ALC measures.

Directory `anonymity_loss_coefficient` is a copy of the code used for both the ALC measure and the attacks, version 1.0.29. This code from this location is not used directly by the experimental code. Rather, the code as used here is pip installed.