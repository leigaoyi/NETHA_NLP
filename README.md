# NETHA_NLP
Running order:


python3 concat.py   % 训练NEZHA模型


python3 adverse.py  %训练Google bert-base 模型


python3 eval.py     % 测试infer.py代码是否正确

## 参考资料
苏剑林大神的[bert4keras](https://github.com/bojone/bert4keras)，版本在0.5.6及以下



## 比赛思路总结
小的提升，可以是参数的微小变化，或者单纯是运气；大的改善，一定是做了对的事情。目前我的进步过程如下：

0.92 --------     调通notebook，使用google bert-base 中文

0.93 --------      max_len 设置为128

0.94 --------     分别尝试了对抗性策略，和梯度补偿策略。目前感觉对抗性策略的最终结果更优。将两种策略结合的效果一般。

0.95 --------      不成熟的交叉验证方法，同一个模型，在五份数据集上运行，存储测试集上最优的模型

0.954 -------       成熟的五折交叉验证，模型在五个数据集上存储五次，最终结果投票，五选三

0.956 -------       两种模型融合，存储七个ckpt，七选四。（注：七选五的测试效果不佳）



## Submission
请参考阿里天池[docker入门介绍](https://tianchi.aliyun.com/competition/entrance/231759/tab/174?spm=5176.12586973.0.0.53c765fdfcp96c)，然后根据自己的镜像地址和打包版本，替换一下代码。

docker pull registry.cn-shanghai.aliyuncs.com/tcc-public/keras:latest-cuda10.0-py3


docker build -t registry.cn-shenzhen.aliyuncs.com/gpu_leigaoyi/tianchi/cy/qoo:5.2 .

docker login --username=\<your user name\>@registry.cn-shenzhen.aliyuncs.com
  
docker push registry.cn-shenzhen.aliyuncs.com/gpu_leigaoyi/tianchi/cy/qoo:5.2

在竞赛平台上提交

## 待办事项
1. albert 和albertV2，可能要换框架

2. 目前的模型都使用的base，没有使用large、xlarge等参数更多的模型。（前期实验效果一般，便专注于bert-base）

3. 单模型的调参，我只到了0.945，还有提升空间。

4. 多模型融合的策略，还可以尝试。目前我尝试了两种模型融合2+5、3+4，成绩最好为0.9566

