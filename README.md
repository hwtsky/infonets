# <p align="center">InfoNets
By Wentao Huang

InfoNets is a Pytorch-based Python package that provides a high-level neural networks API and the efficiency algorithm for deep learning which is not based on the traditional BP (Back Propagation) algorithm. 

TensorFlow, Pytorch and other opensource deep learning software frameworks provide great convenience for current deep learning researchers and developers, and make it easy to study and develop deep learning related algorithms and products. However, they only provide good encapsulation and support for traditional deep learning that is based on BP (Back Propagation) algorithm, and it is not very convenient for non-BP-based deep learning method. 

Based on this, I developed this framework and software package, which can well support the non-BP-based deep learning method, and also integrate the BP algorithm into the same framework at the same time. 

 Google的TensorFlow、Facebook的Pytorch等开源深度学习软件框架为当前深度学习研究与开发人员提供了极大的方便，
让研究与开发深度学习相关算法及产品变得很容易，能让研发的算法与程序很方便地运行于CPU及GPU上。
但它们一个很大的缺陷就是只对传统的基于BP（反向传播）算法的深度学习提供了很好的封装与支持，
对那些不是基于BP算法的深度学习与机器学习方法的开发人员并不是很方便。基于此，我开发了这套基于Python及Pytorch的
框架及软件包，能很好地支持非BP算法的深度学习及机器学习的开发，并能将BP算法同时很好地整合进同一个框架中，
对一般数据集的操作与管理也提供了更强大的支持，让在此软件包基础上进行二次开发变得非常容易。
同时将本人研发的非传统基于BP算法的深度学习模型与方法高度整合进本框架，该方法基于信息理论与脑神经启发，
能自下而上地进行逐层快速地训练一个深度神经网络，并能同时对大批量的训练样本进行并行处理，非常适合GPU的运算，
能让训练深度网络的效率与效果得到极大的提升，而且基于此软件包再来进行二次开发一些相关的应用将变得很容易。
