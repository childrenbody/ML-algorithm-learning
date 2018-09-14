# ML-algorithm-learning
### 机器学习中的算法学习
+ 参考文献
    + 《统计学习方法》- 李航
--------------------------------
+ 目录
    + [1. 决策树](###1)
        + [1.1 CART算法](####1.1)
--------------------------------
<h3 id='1'>1. 决策树</h3>
<h4 id='1.1'>1.1 CART算法</h4>
CART是在给定输入随机变量*X*条件下输出随机变量*Y*的条件概率分布的学习方法。
CART假设决策树是二叉树，内部结点特征的取值为“是”和“否”，左分支是取值为“是”的分支，右分支是取值为“否”的分支。这样的决策树等价与递归地二分每个特征，将输入空间即特征空间划分为有限个单元，并在这些单元上确定预测的概率分布，也就是在输入给定的条件下输出的条件概率分布。

+ CART算法由以下两步组成：
    + （1）决策树生成：基于训练数据集生成决策树，生成的决策树要尽量大；
    + （2）决策树剪枝：用验证数据集对已生成的树进行剪枝病选择最优子树，这时用损失函数最小作为剪枝的标准。

##### CART生成
决策树的生成就是递归地构建二叉决策树的过程，对回归树用平方误差最小化准则，对分类树用基尼指数（Gini index）最小化准则，进行特征选择，生成二叉树。

**1. 分类树的生成**

分类树用基尼指数选择最优特征，同时决定该特征的最优二值切分点。

**定义（基尼指数）**
分类问题中，假设有*K*个类，样本点属于第*k*类的概率为<a href="https://www.codecogs.com/eqnedit.php?latex=p_k" target="_blank"><img src="https://latex.codecogs.com/png.latex?p_k" title="p_k" /></a>，则概率分布的基尼指数定义为
<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=$Gini(p)=\sum_{k=1}^K{p_k(1-p_k)}=1-\sum_{k=1}^K{p_k^2}$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?$Gini(p)=\sum_{k=1}^K{p_k(1-p_k)}=1-\sum_{k=1}^K{p_k^2}$$" title="$$Gini(p)=\sum_{k=1}^K{p_k(1-p_k)}=1-\sum_{k=1}^K{p_k^2}$$" /></a></p>
对于二类分类问题，若样本点属于第一个类的概率是*p*，则概率分布的基尼指数为
<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=$$Gini=(p)=2p(1-p)$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?$$Gini=(p)=2p(1-p)$$" title="$$Gini=(p)=2p(1-p)$$" /></a></p>
对于给定的样本集合*D*，其基尼指数为
<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=$$Gini(D)=1-\sum_{k=1}^K{\left(\frac{|C_k|}{|D|}\right)^2}$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?$$Gini(D)=1-\sum_{k=1}^K{\left(\frac{|C_k|}{|D|}\right)^2}$$" title="$$Gini(D)=1-\sum_{k=1}^K{\left(\frac{|C_k|}{|D|}\right)^2}$$" /></a></p>
这里，<a href="https://www.codecogs.com/eqnedit.php?latex=C_k" target="_blank"><img src="https://latex.codecogs.com/png.latex?C_k" title="C_k" /></a>是*D*中属于第*k*类的样本子集，*K*是类的个数。如果样本集合*D*根据特征*A*是否取某一可能值*a*被分割成<a href="https://www.codecogs.com/eqnedit.php?latex=D_1" target="_blank"><img src="https://latex.codecogs.com/png.latex?D_1" title="D_1" /></a>和<a href="https://www.codecogs.com/eqnedit.php?latex=D_2" target="_blank"><img src="https://latex.codecogs.com/png.latex?D_2" title="D_2" /></a>两部分，即
<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=D_1=\{(x,y)\in{D}|A(x)=a\},D_2=D-D_1" target="_blank"><img src="https://latex.codecogs.com/png.latex?D_1=\{(x,y)\in{D}|A(x)=a\},D_2=D-D_1" title="D_1=\{(x,y)\in{D}|A(x)=a\},D_2=D-D_1" /></a></p>
则在特征*A*的条件下，集合*D*的基尼指数定义为
<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=Gini(D,A)=\frac{|D_1|}{|D|}Gini(D_1)&plus;\frac{|D_2|}{|D|}Gini(D_2)" target="_blank"><img src="https://latex.codecogs.com/png.latex?Gini(D,A)=\frac{|D_1|}{|D|}Gini(D_1)&plus;\frac{|D_2|}{|D|}Gini(D_2)" title="Gini(D,A)=\frac{|D_1|}{|D|}Gini(D_1)+\frac{|D_2|}{|D|}Gini(D_2)" /></a></p>
基尼指数Gini(*D*)表示集合*D*的不确定性，基尼指数Gini(*D*,*A*)表示经*A=a*分割后集合*D*的不确定性。基尼指数值越大，样本集合的不确定性也就越大，这一点与熵相似。

**CART生成算法**

输入：训练数据集*D*，停止计算的条件；

输出：CART决策树.

根据训练数据集，从根节点开始，递归地对每个节点进行一下操作，构建二叉决策树：

（1）设节点的训练数据集为*D*，计算现有特征对该数据集的基尼指数。此时，对每一个特征*A*，对其可能取的每个值*a*，根据样本点对*A=a*的测试为“是”或“否”将*D*分割成<a href="https://www.codecogs.com/eqnedit.php?latex=D_1" target="_blank"><img src="https://latex.codecogs.com/png.latex?D_1" title="D_1" /></a>和<a href="https://www.codecogs.com/eqnedit.php?latex=D_2" target="_blank"><img src="https://latex.codecogs.com/png.latex?D_2" title="D_2" /></a>两部分，利用
<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=Gini(D,A)=\frac{|D_1|}{|D|}Gini(D_1)&plus;\frac{|D_2|}{|D|}Gini(D_2)" target="_blank"><img src="https://latex.codecogs.com/png.latex?Gini(D,A)=\frac{|D_1|}{|D|}Gini(D_1)&plus;\frac{|D_2|}{|D|}Gini(D_2)" title="Gini(D,A)=\frac{|D_1|}{|D|}Gini(D_1)+\frac{|D_2|}{|D|}Gini(D_2)" /></a></p>
计算*A=a*时的基尼指数.

（2）在所有可能的特征*A*以及它们所有可能的切分点*a*中，选择基尼指数最小的特征及其对应的切分点作为最优特征与最优切分点，从现节点生成两个子节点，将训练数据集依特征分配到两个子节点中去。

（3）对两个子节点递归调用（1），（2），直至满足停止条件。

（4）生成CART决策树

算法停止计算的条件是结点中的样本个数小于预定阈值，或样本集的基尼指数小于预定阈值（样本基本属于同一类），或者没有更多的特征。

**代码**

这个代码全用的pandas中dataframe自带的数据分析函数实现，没用numpy中的数组之类的，这个性能就差了些，但是熟悉了dataframe各种函数的用法，有失有得。

先定义一个二叉树用来保存待生成的CART决策树，yes分支代表，当前该特征是否等于该取值：node=value，no分支则代表node！=value

```
class Tree:
    '''用于保存决策数信息的二叉树'''
    def __init__(self, node, value, gini):
        self.node = node
        self.value = value
        self.yes = None
        self.no = None
        self.gini = gini
```
该函数首先计算
<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=$Gini(p)=\sum_{k=1}^K{p_k(1-p_k)}=1-\sum_{k=1}^K{p_k^2}$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?$Gini(p)=\sum_{k=1}^K{p_k(1-p_k)}=1-\sum_{k=1}^K{p_k^2}$$" title="$$Gini(p)=\sum_{k=1}^K{p_k(1-p_k)}=1-\sum_{k=1}^K{p_k^2}$$" /></a></p>

然后计算“是”或“否”的集合
<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=\frac{|D_1|}{|D|}Gini(D_1)\quad&space;or\quad\frac{|D_2|}{|D|}Gini(D_2)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\frac{|D_1|}{|D|}Gini(D_1)\quad&space;or\quad\frac{|D_2|}{|D|}Gini(D_2)" title="\frac{|D_1|}{|D|}Gini(D_1)\quad or\quad\frac{|D_2|}{|D|}Gini(D_2)" /></a></p>

```
def calc_gini(x, total):
    '''计算基尼指数'''
    res = 0
    temp = x[label].value_counts()
    for k in temp.index:
        res += (temp[k] / x.shape[0])**2
    return (1 - res) * x.shape[0] / total
```
遍历数据集计算每个特征的每个取值，返回一个字典，该字典存有每个特征及其取值的基尼指数
```
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
```
选择基尼指数最小的特征及其切分点，从现节点分成两个子节点，将样本集合分配到两个子节点中去，递归调用该函数直到无特征可划分
```
def create_tree(data, label):
    # 如果无特征可分，则取数量最多的类别
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
        # 如果该特征在这个分支下没有样本点，则取数量最多的类别作为叶子节点
        node.yes = data[label].value_counts().idxmax()        
    if not no.empty:
        if no[label].nunique() == 1:
            node.no = no[label].unique()[0]
        else:
            node.no = create_tree(no, label)
    else:
        node.no = data[label].value_counts().idxmax()        
    return node
```
预测数据：遍历决策树，找到符合条件的叶子节点
```
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
```