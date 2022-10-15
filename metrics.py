import torch
import pandas as pd
import numpy as np

def get_df(sensitive_attributes, output, label):
    df_dict = {}
    df_dict["sensitive"] = sensitive_attributes
    df_dict['y_hat'] = list(output)
    df_dict['y'] = list(label)
    df = pd.DataFrame.from_dict(df_dict)
    return df

def accuracy(output, label):
    return np.mean(np.array(output) == np.array(label))

def demographic_parity_violation_binary(sensitive_attributes, output, label):
    df = get_df(sensitive_attributes, output, label)
    positive_value_rates = []
    for attr in set(sensitive_attributes):
        positive_value_rates.append(len(df.loc[(df['sensitive'] == attr) & (df['y_hat'] == 1)].index)/len(df.loc[df['sensitive'] == attr].index))
    positive_value_rates = sorted(positive_value_rates)
    maximum_diff = positive_value_rates[-1] - positive_value_rates[0]
    return maximum_diff

def equalized_odds_violation_binary(sensitive_attributes, output, label):
    df = get_df(sensitive_attributes, output, label)
    true_positive_rates = []
    false_positive_rates = []
    for attr in set(sensitive_attributes):
        true_positive_rates.append(len(df.loc[(df['sensitive'] == attr) & (df['y_hat'] == 1) & (df['y'] == 1)].index)/len(df.loc[(df['sensitive'] == attr) & (df['y'] == 1)].index))
        false_positive_rates.append(len(df.loc[(df['sensitive'] == attr) & (df['y_hat'] == 1) & (df['y'] == 0)].index)/len(df.loc[(df['sensitive'] == attr) & (df['y'] == 0)].index))
    
    true_positive_rates = sorted(true_positive_rates)
    false_positive_rates = sorted(false_positive_rates)

    max_diff_tpr = true_positive_rates[-1] - true_positive_rates[0]
    max_diff_fpr = false_positive_rates[-1] - false_positive_rates[0]
    return max(max_diff_tpr, max_diff_fpr)

def demographic_parity_violation_multiple(sensitive_attributes, output, label):
    total_unique_labels = set(label)
    demographic_parity = 0
    for i in total_unique_labels:
        labels_temp = (torch.tensor(label) == i).float().tolist()
        output_temp = (torch.tensor(output) == i).float().tolist()
        demographic_parity = max(demographic_parity, demographic_parity_violation_binary(sensitive_attributes, output_temp, labels_temp))
    return demographic_parity

def equalized_odds_violation_multiple(sensitive_attributes, output, label):
    total_unique_labels = set(label)
    equalized_odds = 0
    for i in total_unique_labels:
        labels_temp = (torch.tensor(label) == i).float().tolist()
        output_temp = (torch.tensor(output) == i).float().tolist()
        equalized_odds = max(equalized_odds, equalized_odds_violation_binary(sensitive_attributes, output_temp, labels_temp))
    return equalized_odds