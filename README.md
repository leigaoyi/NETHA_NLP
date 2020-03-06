# NETHA_NLP
Running order:
python3 concat.py   % 训练NEZHA模型
python3 adverse.py  %训练Google bert-base 模型
python3 eval.py     % 测试infer.py代码是否正确

## Submission
docker pull registry.cn-shanghai.aliyuncs.com/tcc-public/keras:latest-cuda10.0-py3


docker build -t registry.cn-shenzhen.aliyuncs.com/gpu_leigaoyi/tianchi/cy/qoo:5.2 .

docker login --username=<your user name> registry.cn-shenzhen.aliyuncs.com
  
docker push registry.cn-shenzhen.aliyuncs.com/gpu_leigaoyi/tianchi/cy/qoo:5.2

在竞赛平台上提交
