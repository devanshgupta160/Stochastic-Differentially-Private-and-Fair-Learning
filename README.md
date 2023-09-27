# Stochastic-Differentially-Private-and-Fair-Learning

For tuning our hyperparameters, please use the args.py file to set the dataset, privacy parameters (epsilon and delta), the range of fairness to accuracy to tradeoff(lambda), learning rates, and the gradient clipping or Lischitz constants. Then, run our code using the command
```
	python3 dp_fermi.py
```

To run the equalized odds version of the code, kindly set the ```demographic_parity = False``` in the args.py file.

To reproduce the exact figures in the plots presented in our paper and save time, we have provided the .pt files with the summarized data that we obtained after training the models. So, kindly run the command
```
	python3 ./Figures/plotting.py
```

The plots can also be reproduced by retraining the models according to the given hyperparameters in the paper. The results obtained in the paper were over 15 runs for each configuration. So, if those results need to be extracted, kindly put the num_models_train flag to a number greater than or equal to 15 for more accurate results. For the large-scale experiments, set the ```num_layers = None``` in the args.py file.
