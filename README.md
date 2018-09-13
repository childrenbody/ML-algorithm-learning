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

<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=$$D_1=\{(x,y)\in{D}|A(x)=a\}$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?$$D_1=\{(x,y)\in{D}|A(x)=a\}$$" title="$$D_1=\{(x,y)\in{D}|A(x)=a\}$$" /></a></p>

