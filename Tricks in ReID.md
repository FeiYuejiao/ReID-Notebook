# Tricks in ReID

[TOC]

### 学习率变化策略（Warm up）

![img](https://pic2.zhimg.com/80/v2-71d46cace2918be9bcdd3d722851c5fd_1440w.jpg)

$$\operatorname{lr}(t)=\left\{\begin{array}{ll}3.5 \times 10^{-5} \times \frac{t}{10} & \text { if } t \leq 10 \\ 3.5 \times 10^{-4} & \text {if } 10<t \leq 40 \\ 3.5 \times 10^{-5} & \text {if } 40<t \leq 70 \\ 3.5 \times 10^{-6} & \text {if } 70<t \leq 120\end{array}\right.$$

##### constant warmup

Warm up是在ResNet论文中提到的一种学习率预热的方法。由于刚开始训练时模型的权重(weights)是随机初始化的，此时选择一个较大的学习率，可能会带来模型的不稳定。Warm up就是在刚开始训练的时候先使用一个较小的学习率，训练一些epoches或iterations，等模型稳定时再修改为预先设置的学习率进行训练。ResNet论文中使用一个110层的ResNet在cifar10上训练时，先用0.01的学习率训练直到训练误差低于80%(大概训练了400个iterations)，然后使用0.1的学习率进行训练。

##### gradual warmup

18年Facebook又针对上面的warmup进行了改进，因为从一个很小的学习率一下变为比较大的学习率可能会导致训练误差突然增大。gradual warmup从最开始的小学习率开始，每个iteration增大一点，直到变成最初设置的比较大的学习率。

```python
from torch.optim.lr_scheduler import _LRScheduler

class GradualWarmupScheduler(_LRScheduler):    
"""     
Args:        
	optimizer (Optimizer): Wrapped optimizer.        
	multiplier: target learning rate = base lr * multiplier        
	total_epoch: target learning rate is reached at total_epoch, gradually        
	after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)    
"""
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):        
    	self.multiplier = multiplier        
      if self.multiplier <= 1.:            
        raise ValueError('multiplier should be greater than 1.')        
        self.total_epoch = total_epoch        
        self.after_scheduler = after_scheduler        
        self.finished = False        
        super().__init__(optimizer)
      
    def get_lr(self):        
      if self.last_epoch > self.total_epoch:            
        if self.after_scheduler:                
          if not self.finished:                    
            self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]                    
            self.finished = True                
            return self.after_scheduler.get_lr()            
          return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
      
    def step(self, epoch=None):        
      if self.finished and self.after_scheduler:            
        return self.after_scheduler.step(epoch)        
      else:            
        return super(GradualWarmupScheduler, self).step(epoch)
```

### Cutout

Cutout是一种正则化方法。原理是在训练时随机把图片的一部分减掉，这样能提高模型的鲁棒性。它的来源是计算机视觉任务中经常遇到的物体遮挡问题。通过cutout生成一些类似被遮挡的物体，不仅可以让模型在遇到遮挡问题时表现更好，还能让模型在做决定时更多地考虑环境(context)。

![image-20200610172055376](https://tva1.sinaimg.cn/large/007S8ZIlgy1gfnbfe481wj30is0l5gnz.jpg)



### Random Erasing

Random erasing和cutout非常类似，是一种模拟物体遮挡情况的数据增强方法。区别在于，cutout是把图片中随机抽中的矩形区域的像素值置为0，相当于裁剪掉，random erasing是用随机数或者数据集中像素的平均值替换原来的像素值。而且，cutout每次裁剪掉的区域大小是固定的，Random erasing替换掉的区域大小是随机的。

这是进行分类问题常见的数据增强方式，对行人检测十分有效！在行人重识别中特别是现实生活中，大部分人体会存在遮挡的情况，为了使模型更具有鲁棒性，我们对每个mini-batch使用REA(Random Erasing Augmentation)，REA随机选择一个大小为 ![[公式]](https://www.zhihu.com/equation?tex=%28w_e%2C+H_e%29) 的正方形区域 ![[公式]](https://www.zhihu.com/equation?tex=I_e) ，然后对这个区域的像素用一些随机值代替。其中擦除区域面积占比0.02< ![[公式]](https://www.zhihu.com/equation?tex=S_e) <0.4,擦除区域长宽比 ![[公式]](https://www.zhihu.com/equation?tex=r_l%3D0.3%2Cr_h%3D3.33) ,随机概率为0.5，具体算法如下：

![image-20200610165615637](https://tva1.sinaimg.cn/large/007S8ZIlgy1gfnapqe134j30gd0codkz.jpg)

![img](https://pic3.zhimg.com/80/v2-52a31adf3ed1c7f90aa8ce0f55500f8e_1440w.jpg)

### Random patch





### Label Smoothing

假设分类问题的网络最后的分类为N，也就是ID的数量。且y为真实的ID标签，而$p_i$为预测的ID分类。则其交叉熵损失为：

$ L(I D)=\sum_{i=1}^{N}-q_{i} \log \left(p_{i}\right)\left\{\begin{array}{l}q_{i}=0, y \neq i \\ q_{i j}=1, y_{j}=i\end{array}\right. $

在分类问题中，最后一层一般是全连接层，对应标签的one-hot编码。这种编码方式和通过降低交叉熵损失来调整参数的方式结合起来，会有一些问题。这种方式会鼓励模型对不同类别的输出分数差异非常大，或者说，模型过分相信它的判断。但是，对于一个由多人标注的数据集，不同人标注的准则可能不同，每个人的标注也可能会有一些错误。模型对标签的过分相信会导致过拟合。

标签平滑(Label-smoothing regularization,LSR)的具体思想是降低我们对于标签的信任，例如将损失的目标值从1稍微降到0.9，或者从0稍微升到0.1。标签平滑最早在inception-v2中被提出，它将真实的概率改造为：

$q_{i}=\left\{\begin{array}{ll}1-\varepsilon & \text { if } i=y \\ \varepsilon /(K-1) & \text { otherwise }\end{array}\right.$

其中，ε是一个小的常数，K是这个类别的数目，y是图片的真正的标签，i代表第i个类别，$q_i$是图片为第i类的概率。

总的来说，LSR是一种通过在标签y中加入噪声，实现对模型约束，降低模型过拟合程度的一种正则化方法。

```python
import torch
import torch.nn as nn

class LSR(nn.Module):
    def __init__(self, e=0.1, reduction='mean'):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e
        self.reduction = reduction

    def _one_hot(self, labels, classes, value=1):
        """
            Convert labels to one hot vectors
        Args:
            labels: torch tensor in format [label1, label2, label3, ...]
            classes: int, number of classes
            value: label value in one hot vector, default to 1
        Returns:
            return one hot format labels in shape [batchsize, classes]
        """
        one_hot = torch.zeros(labels.size(0), classes)
        #labels and value_added  size must match
        labels = labels.view(labels.size(0), -1)
        value_added = torch.Tensor(labels.size(0), 1).fill_(value)
        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)
        one_hot.scatter_add_(1, labels, value_added)
        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        """
        		convert targets to one-hot format, and smooth them.
        Args:
            target: target in form with [label1, label2, label_batchsize]
            length: length of one-hot format(number of classes)
            smooth_factor: smooth factor for label smooth
        Returns:
            smoothed labels in one hot format
        """
        one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        one_hot += smooth_factor / length
        return one_hot.to(target.device)
```

### Last Stride

相关的论文发现在backbone中移除最后一个将采样的操作可以丰富特征的细粒度，通常而言，增大尺寸一般是能够提升性能的。作者为了简单方便，直接把ResNet-50的最后一个卷积层的Stride由2变成了1，当输入图片为256x128时，最后一层输出的feature map的尺寸为16x8，而不是原来的8x4。

### BNNeck

在行人重识别模型中，有很多工作都是融合了ID loss和Triplet loss来进行训练的，但是这种loss函数的目标并不协调，对于ID loss，特别在行人重检测，consine距离比欧氏距离更加适合作为优化标准，而Triplet loss更加注重在欧式空间提高类内紧凑性和类间可分性。因此两者关注的度量空间不一致，这就会导致一个可能现象就是当一个loss减小时另外一个在震荡或增大。因此作者设计了BNNeck用于解决这个问题，如下图：

![image-20200610192340249](https://tva1.sinaimg.cn/large/007S8ZIlgy1gfnez4uldkj30hc0mgdlf.jpg)

通过神经网络提取特征ft用于Triplet loss，然后通过一个BN层变成了fi，在训练时，分别使用ft和fi来优化这个网络，由于BN层的加入，ID loss就更容易收敛，另外，BNNeck也减少了ID loss对于ft优化的限制，从而使得Triplet loss也变得容易收敛了。因为超球体几乎是对称的坐标轴的原点，BNNECK的另一个技巧是去除分类器fc层的偏差，这个偏差会限制分类的超球面。在测试时，使用fi作为ReID的特征提取结果，这是由于Cosine距离相比欧氏距离更加有效的原因。

### Center Loss

Triplet loss有个缺点是只考虑了相对距离，其loss大小与正样本对的绝对距离无关。举个例子，假如margin=0.3。正样本对距离是1.0，负样本对是1.1，最后loss是0.2。正样本对距离是2.0，负样本对是2.1，最后loss还是0.2。为了增加正样本之间的聚类性能，我们加入了Center loss：

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BL%7D_%7BC%7D%3D%5Cfrac%7B1%7D%7B2%7D+%5Csum_%7Bj%3D1%7D%5E%7BB%7D%5Cleft%5C%7C%5Cboldsymbol%7Bf%7D_%7Bt_%7Bj%7D%7D-%5Cboldsymbol%7Bc%7D_%7By_%7Bj%7D%7D%5Cright%5C%7C_%7B2%7D%5E%7B2%7D+%5C%5C)

由于ReID现在的评价指标主要是cmc和mAP，这两个都是检索指标，所以center loss可能看上效果不是那么明显。但是center loss会明显提高模型的聚类性能，这个聚类性能在某些任务场景下是有应用的。比如直接卡阈值区分正负样本对tracking任务。

当然center loss有个小trick就是，更新网络参数和更新center参数的学习率是不一样的，细节需要去看代码，很难说清楚。





### 参考文献

[1] [图像分类任务中的tricks总结](https://mp.weixin.qq.com/s?__biz=MzI4MjA0NDgxNA==&mid=2650722499&idx=1&sn=b489bb77ba12be14df197fdc77893b22&scene=21#wechat_redirect)

[2] [一个更加强力的ReID Baseline](https://zhuanlan.zhihu.com/p/61831669)

[3] [Bag of Tricks and A Strong Baseline for Deep Person Re-identification](http://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf)