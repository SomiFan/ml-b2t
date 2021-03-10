# machine learning with Li Hongyi

## Optimization for DL

### What you have known before?

- SGD
- SGD with momentum
- Adagrad
- RMSProp
- Adam

### Notations

![image-20210306104758058](/home/vince/snap/typora/33/.config/Typora/typora-user-images/image-20210306104758058.png)

### On-line vs Off-line

on-line: 一个time step只能获得一个训练样本计算出y，然后求得Loss

off-line:一个time step可以把所有训练样本放入模型，得出y计算Loss，这是理想情况，之后所有讨论都是基于这个假设

![image-20210306105805571](/home/vince/snap/typora/33/.config/Typora/typora-user-images/image-20210306105805571.png)

### SGD, stochastic gradient descent

每个time step往梯度的反方向走

### SGDM, SGD with Momentum

每个time step的movement都要加一个momentum向量，来避免梯度消失

![image-20210306112034248](/home/vince/snap/typora/33/.config/Typora/typora-user-images/image-20210306112034248.png)

![image-20210306112121612](/home/vince/snap/typora/33/.config/Typora/typora-user-images/image-20210306112121612.png)

momentum即上一步的movement，当某一点梯度接近零时，由于有momentum加入，移动得不会特别慢

### Adagrad

![image-20210306112704699](/home/vince/snap/typora/33/.config/Typora/typora-user-images/image-20210306112704699.png)

learning rate要除以之前所有梯度的平方和的开方，这样假如之前梯度特别大，下降的步子就会小一些（陡峭的地方不要冲得太猛），假如之前梯度特别小，下降的步子就会大一些（平缓的地方走的快些）

### RMSProp

![image-20210306113242492](/home/vince/snap/typora/33/.config/Typora/typora-user-images/image-20210306113242492.png)

避免Adagrad中learning rate要除的那个数无止境增大，引入alpha，对上一步的梯度和之前梯度的平方和做一个平均。

### Adam

以上方法还是没法解决在梯度为零处停止的问题。

Adam其实是SGDM加上RMSProp得到的方法：

- mt代替梯度：mt由上一步的momentum和梯度加权平均，而真正用到的mt_hat，是把mt除以小于1的一个参数，来保证它在t小时足够大
- vt用来修正learning rate，vt同样除以小于1的一个数来保证在t小时足够大，vt_hat加上一个epsilon来保证一开始t=0的时候不会因为vt_hat=0，而使learning rate爆掉。

![image-20210306160601505](/home/vince/snap/typora/33/.config/Typora/typora-user-images/image-20210306160601505.png)

### 为什么14年之后没有更好的optimizer被提出来？

![image-20210306161610936](/home/vince/snap/typora/33/.config/Typora/typora-user-images/image-20210306161610936.png)

近年最优秀的模型几乎都是用Adam和SGDM训练出来的，Adam在训练集上表现最好，但是在validation set上SGDM表现最好，而且SGDM收敛得更好。

![image-20210306162208558](/home/vince/snap/typora/33/.config/Typora/typora-user-images/image-20210306162208558.png)

有人提出开始用Adam，后期用SGDM，但上中间切换算法时的规则很难制定。

### 改进Adam

![image-20210306163051942](/home/vince/snap/typora/33/.config/Typora/typora-user-images/image-20210306163051942.png)

假设前1000步的梯度都是1，突然有一步梯度达到了10000，这一步的learning rate由于前面的影响也只能达到10倍根号10的learning rate，这是adam的问题，在最后阶段，大部分梯度都极小，偶尔一些包含重要信息的大梯度也没法产生作用。

![image-20210306164130547](/home/vince/snap/typora/33/.config/Typora/typora-user-images/image-20210306164130547.png)

假如t-1的v_hat比t步算得的v大的话，t步的v_hat直接取上一步的v_hat，这样，假如有一步梯度特别大，那么后面的learning rate会很小，就凸显了大梯度的作用。但是这样做的话，learning rate会一直减小。只解决了小梯度得到大learning rate的问题是AMSGrad的问题

![image-20210306165115439](/home/vince/snap/typora/33/.config/Typora/typora-user-images/image-20210306165115439.png)

AdaBound给learning rate做了一个clip，给它设计了一个上界upper bound和一个下界lower bound。这样做失去了learning rate的自适应性，所以也不是最好的方法。

### 改进SGDM

![image-20210306170037647](/home/vince/snap/typora/33/.config/Typora/typora-user-images/image-20210306170037647.png)

SGD类的算法不是自适应的，设置小learning rate就走得太慢，设置大learning rate就走得太快。怎么样找到最佳learning rate呢？

![image-20210306170716616](/home/vince/snap/typora/33/.config/Typora/typora-user-images/image-20210306170716616.png)

让learning rate在上下界之间反复循环，learning rate变大时，就是鼓励算法去探索，以更快地收敛。

![image-20210306170949620](/home/vince/snap/typora/33/.config/Typora/typora-user-images/image-20210306170949620.png)

learning rate只做一个cycle，lr最开始一直上升（warm-up），直到到达比较好的local minima，就开始下降（annealing），最后进入微调直至收敛（fine- tuning）。

