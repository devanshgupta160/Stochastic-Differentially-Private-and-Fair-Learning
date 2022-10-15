Stochastic-Differentially-Private-and-Fair-Learning

For tuning our hyperparameters, please use the args.py file to accordingly set the dataset, privacy parameters, lambda list, and other hyperparameters. Then, run our code using the command
	python3 dp_fermi.py

Note that to run the equalized odds version of the code, kindly set the demographic_parity flag in the args.py file to be False.

To reproduce the exact figures in the plots and save time, we have provided the .pt files with the summarized data that we obtained after training the models. So, kindly run the command 
	python3 ./Figures/plotting.py

The plots can also be reproduced by retraining the models according to the given hyperparameters in the paper. But the results that we obtained were over 15 runs for each configuration. So, if those results need to be extracted, kindly put the num_models_train flag to a number greater than or equal to 15 for more accurate results. For the large scale experiments, set the num_layers flag to be None in the args.py file.
