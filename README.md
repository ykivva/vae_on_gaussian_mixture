# VAE on Gaussian Mixture

## Quick start

To train a vae model you need to make the following steps:

1. Add configuration for training to the dict ```vae_config``` in the ```config.py```
   1.  ```data_dim``` - dimension of the generated data
   2.  ```latent_dim``` -  dimension of the latent variable ```z```
   3.  ```std_grad``` - **true** if std is not known and we need to learn it, otherwise **false**
   4.  ```epochs``` - how many times we will iterate through the dataset
   5.  ```batch_size``` - size of the batch
   6.  ```alphas``` - list of ratios between number of samples in dataset and data dimension
   7.  ```std``` - std of the generated data
   8.  ```p_bernoulli``` - probability that point belongs to the first cluster
   9.  ```d``` - rescale parameter for the gaussian mixture centroids
2. Run the command:
 
```python -m train --vae_model_name=<name_of_vae_model> --vae_dataset_name=<name_of_dataset> --train_config=<name_of_train_config>```
