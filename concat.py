# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 18:34:30 2020

@author: kasy
"""

#! -*- coding:utf-8 -*-
# 通过对抗训练增强模型的泛化性能
# 比CLUE榜单公开的同数据集上的BERT base的成绩高2%
# 数据集：IFLYTEK' 长文本分类 (https://github.com/CLUEbenchmark/CLUE)
# 博客：https://kexue.fm/archives/7234

import json
import numpy as np
from bert4keras.backend import set_gelu
from bert4keras.backend import keras, search_layer, K
from bert4keras.tokenizer import Tokenizer
from bert4keras.bert import build_bert_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import Lambda, Dense
from tqdm import tqdm
from keras.layers import *

import pandas as pd

#set_gelu('tanh')  # 切换gelu版本


epoch_num = 18
prefix = 'NEZHA'


num_classes = 2
maxlen = 128
batch_size = 64
lr = 1.5e-5

alpha = 0.4 # 对抗性权重

## BERT base
#config_path = 'publish/bert_config.json'
#checkpoint_path = 'publish/bert_model.ckpt'
#dict_path = 'publish/vocab.txt'

# HuaWei NeTha
config_path = 'NEZHA/bert_config.json'
checkpoint_path = 'NEZHA/model.ckpt-900000'
dict_path = 'NEZHA/vocab.txt'


def load_data(filename):
    D = pd.read_csv(filename).values.tolist()
    return D


## 加载数据集
#all_data = load_data('./data/train_20200228.csv')
#random_order = range(len(all_data))
#np.random.shuffle(list(random_order))
##train_data = [all_data[j] for i, j in enumerate(random_order) if i % 6 != 1 and i%6!=2]
##valid_data = [all_data[j] for i, j in enumerate(random_order) if i % 6 == 1]
##test_data = [all_data[j] for i, j in enumerate(random_order) if i % 6 == 2]
#
#train_data = [all_data[j] for i, j in enumerate(random_order) if i % 5 != 1]
#valid_data = [all_data[j] for i, j in enumerate(random_order) if i % 5 == 1]
##test_data = [all_data[j] for i, j in enumerate(random_order) if i % 6 == 2]

# 建立分词器

#val_test_data = load_data('./data/dev_20200228.csv')

tokenizer = Tokenizer(dict_path, do_lower_case=True)




class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for i in idxs:
            #print(self.data[i])
            _,_, text1, text2, label = self.data[i]
#             print(text1, text2, label)
            token_ids, segment_ids = tokenizer.encode(text1, text2, max_length=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


## 转换数据集
#train_generator = data_generator(train_data, batch_size)
#valid_generator = data_generator(valid_data, batch_size)
#val_test_generator = data_generator(val_test_data, batch_size)

## 加载预训练模型
#bert = build_bert_model(
#    config_path=config_path,
#    checkpoint_path=checkpoint_path,
#    return_keras_model=False,
#)

#bert = build_bert_model(
#    config_path=config_path,
#    checkpoint_path=checkpoint_path,
#    with_pool=True,
#    return_keras_model=False,
#)
                
##加载预训练模型:: 华为
bert = build_bert_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='nezha',
    with_pool=True,
    return_keras_model=False,
)                

output = Dropout(rate=0.1)(bert.model.output)
## 加了adversarial 层后，可以考虑更稳定些
#output = Lambda(lambda x: x[:, 0])(bert.model.output)

output = Dense(units=2,
               activation='softmax',
               kernel_initializer=bert.initializer)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()


##adding gradient panalty
#def sparse_categorical_crossentropy(y_true, y_pred):
#    """自定义稀疏交叉熵
#    这主要是因为tf自带的sparse_categorical_crossentropy不支持求二阶梯度。
#    """
#    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
#    y_true = K.cast(y_true, 'int32')
#    y_true = K.one_hot(y_true, K.shape(y_pred)[-1])
#    return K.mean(K.categorical_crossentropy(y_true, y_pred))
#
#
#def loss_with_gradient_penalty(y_true, y_pred):
#    """带梯度惩罚的loss
#    """
#    loss = sparse_categorical_crossentropy(y_true, y_pred)
#    embeddings = search_layer(y_pred, 'Embedding-Token').embeddings
#    gp = K.sum(K.gradients(loss, [embeddings])[0].values**2)
#    return loss + 0.5 * gp


#model.compile(
#    loss=loss_with_gradient_penalty,
#    optimizer=Adam(lr),  # 用足够小的学习率
#    #optimizer=PiecewiseLinearLearningRate(Adam(5e-5), {10000: 1, 30000: 0.1}),
#    metrics=['accuracy'],
#)


model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(lr),
    metrics=['accuracy'],
)


def adversarial_training(model, embedding_name, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (model._feed_inputs +
              model._feed_targets +
              model._feed_sample_weights)  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数


# 写好函数后，启用对抗训练只需要一行代码
adversarial_training(model, 'Embedding-Token', alpha)



def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self, best_val_acc=0., num=0):
        self.best_val_acc = best_val_acc
        self.num = int(num)

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(test_generator)
        if val_acc > self.best_val_acc:
            print('Model ckpt store!')
            self.best_val_acc = val_acc
            model.save_weights('{0}_best_{1}_model.weights'.format(prefix, self.num))
        #test_acc = evaluate(valid_generator)
        test_acc = val_acc
        print(u'test_acc: %.5f, best_test_acc: %.5f, val_acc: %.5f\n'
              % (val_acc, self.best_val_acc, test_acc))


#========================Init=============================
#print('****************Start init*******************')
#model.save_weights('{0}_best_{1}_model.weights'.format(prefix, 0))

print('****************Start init*******************')
all_data = load_data('./data/small.csv')
random_order = range(len(all_data))
np.random.shuffle(list(random_order))
        
train_data = all_data
valid_data = all_data
test_data = all_data
# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)

evaluator = Evaluator(num=0)
model.fit_generator(train_generator.forfit(),
                    steps_per_epoch=len(train_generator),
                    epochs=2,
                    callbacks=[evaluator])

print('Random init over!')
#================================First==================
#print('**************First Model**********************')
all_data = load_data('./data/train.csv')
random_order = range(len(all_data))
np.random.shuffle(list(random_order))


for turn in range(1, 3):
    print('*****************Turn {}**********************'.format(turn))
    model.load_weights('{0}_best_0_model.weights'.format(prefix))
    train_data = [all_data[j] for i, j in enumerate(random_order) if i % 2 != (turn-1)]
    valid_data = [all_data[j] for i, j in enumerate(random_order) if i % 2 == (turn-1)]
    test_data = valid_data
    # 转换数据集
    train_generator = data_generator(train_data, batch_size)
    valid_generator = data_generator(valid_data, batch_size)
    test_generator = data_generator(test_data, batch_size)
    
    evaluator = Evaluator(num=turn)
    model.fit_generator(train_generator.forfit(),
                        steps_per_epoch=len(train_generator),
                        epochs=epoch_num,
                        callbacks=[evaluator])

    best_score = evaluate(test_generator)
    with open('{}_record_acc.txt'.format(prefix), 'a+') as f:
        f.write('Turn {0} Best acc {1:.4f}\n'.format(turn, best_score))