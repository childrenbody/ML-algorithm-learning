#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 23:14:59 2018

@author: childrenbody
"""
import pandas as pd
import numpy as np

class Tree:
    '''用于保存决策数信息的二叉树'''
    def __init__(self, node, value, gini):
        self.node = node
        self.value = value
        self.yes = None
        self.no = None
        self.gini = gini

def calc_gini(x, total):
    '''计算基尼指数'''
    res = 0
    temp = x[label].value_counts()
    for k in temp.index:
        res += (temp[k] / x.shape[0])**2
    return (1 - res) * x.shape[0] / total

def create_gini_dict(data, label):
    '''计算每一个特征的每一个取值的基尼指数'''
    node_gini = dict()
    feature = [c for c in data.columns if c not in [label]]
    total = data.shape[0]
    for c in feature:
        for i in data[c].unique():
            temp = data[[label]].groupby(data[c] == i).apply(calc_gini, total=total)
            node_gini[(c, i)] = temp.sum()
    return node_gini

# 选择基尼指数最小的特征及其切分点，从现节点分成两个子节点
# 将样本集合分配到两个子节点中去，递归调用该函数直到无特征可划分
def create_tree(data, label):

    if data.shape[1] < 2:
        return data[label].value_counts().idxmax()        
    node_gini = create_gini_dict(data, label)
    c, a = min(node_gini, key=node_gini.get)
    node = Tree(c, a, node_gini[(c, a)])
    yes = data[data[c] == a]
    yes = yes.drop(c, axis=1)
    no = data[data[c] != a]
    if not yes.empty:
        if yes[label].nunique() == 1:
            node.yes = yes[label].unique()[0]
        else:
            node.yes = create_tree(yes, label)
    else:
        node.yes = data[label].value_counts().idxmax()        
    if not no.empty:
        if no[label].nunique() == 1:
            node.no = no[label].unique()[0]
        else:
            node.no = create_tree(no, label)
    else:
        node.no = data[label].value_counts().idxmax()        
    return node

# 预测，遍历二叉树，查找到符合条件的叶子节点
def predict(tree, data):
    res = []
    for i, row in data.iterrows():
        node = tree
        flag = False
        while not flag:
            if row[node.node] == node.value:
                if isinstance(node.yes, Tree):
                    node = node.yes
                else:
                    res.append(node.yes)
                    flag = True
            else:
                if isinstance(node.no, Tree):
                    node = node.no
                else:
                    res.append(node.no)
                    flag = True
    return res


file_path = 'mushrooms_new.csv'
data = pd.read_csv(file_path)
label = 'class'
feature = [c for c in data.columns if c not in [label]]

# 划分训练集和测试集，这里取20%作为测试集
random = list(range(data.shape[0]))
np.random.shuffle(random)
train_size = int(data.shape[0] * 0.8)
train = data.iloc[random[:train_size], :]
test = data.iloc[random[train_size:], :]

tree = create_tree(train, label)
result = predict(tree, test[feature])
print('CART accuracy: {}'.format(sum(test[label] == result)/len(result)))