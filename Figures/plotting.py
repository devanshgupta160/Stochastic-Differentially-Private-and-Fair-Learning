import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import scipy.stats as st
import pickle

plt.rcParams.update({'font.size': 15})

# Plotting Code for Demographic Parity Experiments
for dataset in ["parkinsons", "retired-adult", "adult", "credit-card"]:
    for eps in [0.5, 1, 3, 9, np.inf]:
        fermi_dict = torch.load(f"./Demographic Parity Experiments/FERMI_Multiple_Models/{dataset}/final_eps_{eps}_logistic-regression.pt")
        new_dict = torch.load(f"./Demographic Parity Experiments/DP-ERM_Mitigation/{dataset}/final_eps_{eps}_old_logistic_regression.pt")

        x = []
        y = []
        y_upper = []
        y_lower = []
        for lambd in new_dict.keys():
            x.append(new_dict[lambd]["misclassification"])
            y.append(new_dict[lambd]["demographic_parity"]["mean"])
            y_upper.append(new_dict[lambd]["demographic_parity"]["upper"])
            y_lower.append(new_dict[lambd]["demographic_parity"]["lower"])
        plt.fill_between(x, y_lower, y_upper, alpha = 0.5)
        plt.plot(x, y, label = "Tran et al. (2021a)")

        u = torch.load(f"./Demographic Parity Experiments/PFLD/{dataset}/{dataset}_eps_{eps}_old_logistic_regression.pt")
        plt.scatter(u["Misclassification Error"], u["Demographic Parity"], label = f"Tran et al. (2021b)", marker = "*", color = 'red', s = 160)

        x = []
        y = []
        y_upper = []
        y_lower = []
        for lambd in fermi_dict.keys():
            x.append(fermi_dict[lambd]["misclassification"])
            y.append(fermi_dict[lambd]["demographic_parity"]["mean"])
            y_upper.append(fermi_dict[lambd]["demographic_parity"]["upper"])
            y_lower.append(fermi_dict[lambd]["demographic_parity"]["lower"])
        plt.fill_between(x, y_lower, y_upper, alpha = 0.5)
        plt.plot(x, y, label = "DP-FERMI")

        handles, labels = plt.gca().get_legend_handles_labels()
        order = [0, 2, 1]

        plt.legend([handles[i] for i in order], [labels[i] for i in order])
        plt.xlabel("Misclassification Error")
        plt.ylabel("Demographic Parity Violation")
        plt.savefig(f"{dataset}_eps_{eps}_demographic_parity.png", bbox_inches = "tight")
        plt.cla()

# Plotting Code for Equalized Odds Experiments
for dataset in ["adult", "credit-card"]:
    for eps in [0.5, 1, 3]:
        fermi_dict = torch.load(f"./Equalized Odds Experiments/FERMI_Multiple_Models/{dataset}/final_eps_{eps}_logistic-regression.pt")
        new_dict = torch.load(f"./Equalized Odds Experiments/Jagielski-Inprocessing/{dataset}/eps_{eps}.pt")

        plt.fill_between(new_dict["misclassificaton"], new_dict["equalized_odds"]["upper"], new_dict["equalized_odds"]["lower"], alpha = 0.5)
        plt.plot(new_dict["misclassificaton"], new_dict["equalized_odds"]["mean"], label = "Jagielski et al. (2019)")

        u = torch.load(f"./Equalized Odds Experiments/PFLD/{dataset}_eps_{eps}_old_logistic_regression.pt")
        plt.scatter(u["Misclassification Error"], u["Equalized Odds"], label = f"Tran et al. (2021b)", marker = "*", color = 'red', s = 160)

        x = []
        y = []
        y_upper = []
        y_lower = []
        for lambd in fermi_dict.keys():
            x.append(fermi_dict[lambd]["misclassification"])
            y.append(fermi_dict[lambd]["equalized_odds"]["mean"])
            y_upper.append(fermi_dict[lambd]["equalized_odds"]["upper"])
            y_lower.append(fermi_dict[lambd]["equalized_odds"]["lower"])
        plt.fill_between(x, y_lower, y_upper, alpha = 0.5)
        plt.plot(x, y, label = "DP-FERMI")

        handles, labels = plt.gca().get_legend_handles_labels()
        order = [0, 2, 1]

        plt.legend([handles[i] for i in order], [labels[i] for i in order])
        plt.xlabel("Misclassification Error")
        plt.ylabel("Equalized Odds Violation")
        plt.savefig(f"{dataset}_{eps}_Equalized_Odds.png", bbox_inches = "tight")
        plt.cla()


# Code for the Large Scale Experiments
base_path = "./UTKFace/"
eps_ = [10, 25, 50, 100, 200]
for eps in eps_:
    fermi_dict = torch.load(f"./UTKFace/final_eps_{eps}_resnet-classifier.pt")
    x = []
    y = []
    y_upper = []
    y_lower = []
    for lambd in fermi_dict.keys():
        x.append(fermi_dict[lambd]["misclassification"])
        y.append(fermi_dict[lambd]["demographic_parity"]["mean"])
        y_upper.append(fermi_dict[lambd]["demographic_parity"]["upper"])
        y_lower.append(fermi_dict[lambd]["demographic_parity"]["lower"])
    plt.fill_between(x, y_lower, y_upper, alpha = 0.5, color = "orange")
    plt.plot(x, y, label = "DP-FERMI", color = "orange")
    plt.xlabel("Misclassification Error")
    plt.ylabel("Demographic Parity Violation")
    plt.legend()
    plt.savefig(f"large_scale_{eps}.jpg", bbox_inches = "tight")
    plt.cla()