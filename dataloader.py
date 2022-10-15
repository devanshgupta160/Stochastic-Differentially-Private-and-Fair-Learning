import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torchvision import transforms

import pandas as pd
import random
import numpy as np
from PIL import Image
import os

#Handling Cases of the General Dataset:
#      Format should be in form of the CSV files with a single output column which is binary("Yes" or "No")
#      Categorical columns should be in strings

class GeneralData():
    def __init__(self, path, sensitive_attributes = None, cols_to_norm = None, split = 0.60, output_col_name = None):
        if not sensitive_attributes:
            raise Exception("No Sensitive Attributes Provided. Please provide one or more")

        if not output_col_name:
            raise Exception("No output column name entered. Please provide one")

        self.output_col_name = output_col_name
        self.sensitive_attributes = sorted(sensitive_attributes)

        self.path = path
        df = pd.read_csv(self.path)

        if self.sensitive_attributes:
            non_sens_attr = sorted(list(set(df.columns).difference(set(self.sensitive_attributes + [output_col_name]))))
        else:
            non_sens_attr = sorted(list(df.columns).difference(set([output_col_name])))

        one_hot_cols =list(set(non_sens_attr).difference(cols_to_norm))
        df = pd.get_dummies(df, columns = one_hot_cols)
        self.non_sens_attr = list(set(df.columns).difference(set(self.sensitive_attributes + [output_col_name])))

        #Splitting Data
        self.df_train = df.sample(frac = split, random_state = 100)
        self.df_test = df.drop(self.df_train.index)

        if cols_to_norm:
            self.mean_train = self.df_train[cols_to_norm].mean()
            self.std_train = self.df_train[cols_to_norm].std()

            for col in cols_to_norm:
                self.df_train[col] = self.df_train[col].apply(lambda x: (x - self.mean_train[col]) / self.std_train[col])
                self.df_test[col] = self.df_test[col].apply(lambda x: (x - self.mean_train[col]) / self.std_train[col])

    def getTrain(self):
        return TabularDataset(self.df_train, self.non_sens_attr, self.sensitive_attributes, output_col_name = self.output_col_name)
    
    def getTest(self):
        return TabularDataset(self.df_test, self.non_sens_attr, self.sensitive_attributes, output_col_name = self.output_col_name)

    def calculateP_s(self, demographic_parity = True):
        if demographic_parity:
            dataset = self.getTrain()
            sens = torch.zeros(dataset.count_attr[0])
            for i in range(dataset.__len__()):
                _, u, _, _ = dataset.__getitem__(i)
                sens += u
            sens /= dataset.__len__()
            return torch.diag(1/(sens)**0.5)
        else:
            dataset = self.getTrain()
            diff_matrices = [torch.zeros(dataset.count_attr[0]), torch.zeros(dataset.count_attr[0])]
            lengths = [0, 0]
            for i in range(dataset.__len__()):
                _, u, lab, _ = dataset.__getitem__(i)
                diff_matrices[lab] += u
                lengths[lab] += 1
            diff_matrices[0] /= lengths[0]
            diff_matrices[1] /= lengths[1]
            return [torch.diag(1/(diff_matrices[0])**0.5), torch.diag(1/(diff_matrices[1])**0.5)]


class TabularDataset(Data.Dataset):
    def __init__(self, df, non_sens_attr, sensitive_attributes, output_col_name):
        self.df = df
        self.sensitive_attributes = sensitive_attributes

        self.one_hot_non_senstive = self.df[non_sens_attr]
        self.sensitive_table = self.df[self.sensitive_attributes]
        self.output = self.df[output_col_name]

        self.count_attr = []
        self.attr_no = {}

        for col_name in self.sensitive_attributes:
            self.attr_no[col_name] = {}
            count = 0
            for col_nam_attr in list(self.df[col_name].unique()):
                self.attr_no[col_name][col_nam_attr] = count
                count += 1
            self.count_attr.append(count)

        for i in range(len(self.count_attr) - 2, -1, -1):
            self.count_attr[i] = self.count_attr[i] * self.count_attr[i+1]
        self.count_attr.append(1)

    def __len__(self):
        return len(self.one_hot_non_senstive.index)

    def __getitem__(self, idx):
        non_sensitive_attributes = np.array(self.one_hot_non_senstive.iloc[idx])
        sensitive_one_hot, sens_ind = self.onehotlookup(self.sensitive_table.iloc[idx])
        label = self.output.iloc[idx]
        if label == "Yes":
            label = 1
        else:
            label = 0
        sensitive_vector = []
        return torch.from_numpy(non_sensitive_attributes), sensitive_one_hot, label, sens_ind

    def onehotlookup(self, df):
        one_hot_vector = torch.zeros(self.count_attr[0])
        index = 0
        for i, attr in enumerate(self.sensitive_attributes):
            index += self.count_attr[i + 1] * self.attr_no[attr][df[attr]]
        one_hot_vector[index] = 1
        return one_hot_vector, index

class UTKFaceDataset(Data.Dataset):
    def __init__(self, sensitive_attribute = "race", split = 0.75, train = True, seed = 100):
        self.sens_count = 0
        if sensitive_attribute == "gender":
            self.sens_count = 2
            self.sens_index = 1
        if sensitive_attribute == "race":
            self.sens_count = 5
            self.sens_index = 2
        self.age_ranges = [(0,10), (10,15), (15,20), (20,25), (25,30), (30,40), (40,50), (50,60), (60,120)]
        if sensitive_attribute == "age":
            self.sens_count = 9
            self.sens_index = 0
        self.transforms = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])
        self.path = "./Datasets/UTKFace/UTKFace/"
        self.image_names = os.listdir(self.path)
        if seed:
            self.image_names = random.Random(seed).shuffle(self.image_names)
        if train:
            self.dataset = self.image_names[:int(split * len(self.image_names))]
        else:
            self.dataset = self.image_names[int(split * len(self.image_names)):]

    def __getitem__(self, idx):
        image_retr = self.dataset[idx]
        attributes = image_retr.split("_")[:-1]

        sensitive_one_hot = torch.zeros(self.sens_count)
        sens_ind = int(attributes[self.sens_index])
        sensitive_one_hot[sens_ind] = 1

        label = 0
        for i, (a, b) in enumerate(self.age_ranges):
            if a < int(attributes[0]) and int(attributes[0]) <= b:
                label = i
                break
        
        image_vec = Image.open(self.path + image_retr)
        u = self.transforms(image_vec)

        return u, sensitive_one_hot, label, sens_ind

    def __len__(self):
        return len(self.dataset)
    
    def calculateP_s(self):
        sens = torch.zeros(self.sens_count)
        for i in range(self.__len__()):
            _, u, _, _ = self.__getitem__(i)
            sens += u
        sens /= self.__len__()
        return torch.diag(1/(sens)**0.5)