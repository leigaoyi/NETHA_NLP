# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 09:32:22 2020

@author: kasy
"""

from keras.layers import *

from bert4keras.backend import keras, set_gelu, K
from bert4keras.backend import search_layer
from bert4keras.bert import build_bert_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.tokenizer import Tokenizer
import pandas as pd
import numpy as np
#from bert4keras.optimizers import PiecewiseLinearLearningRate

#import re
#def reproduce_text(text):
#    ch_reg = "[\u002c\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b]"
#
#    new_test = re.sub(ch_reg, '，', text)
#    if (new_test[-1] == '，') or (new_test[-1] == '?'):
#        #print('-- '+new_test[:-1])
#        new_test = new_test[:-1] 
#    else:
#        new_test = new_test
#    return new_test

import re

ch_reg = "[\u002c\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\u300a\u300b]"

f = lambda x:re.sub(ch_reg, '，', x).replace(ch_reg, '，').strip('，')

#def load_data(filename):
#    D = pd.read_csv(filename)
#    D['query1'] = D['query1'].map(f)
#    D['query2'] = D['query2'].map(f)
#    D = D.values.tolist()
#    return D

def load_data(filename):
    D = pd.read_csv(filename).values.tolist()
    return D

#set_gelu('tanh')  # 切换gelu版本
## Penalty + google : 

epoch_num = 8
prefix = 'NEZHA'

lr = 1.5e-5

maxlen = 128
batch_size = 64


# HuaWei NeTha
config_path = 'NEZHA/bert_config.json'
checkpoint_path = 'NEZHA/model.ckpt-900000'
dict_path = 'NEZHA/vocab.txt'

print(config_path)


# 加载数据集

#train_data = [all_data[j] for i, j in enumerate(random_order) if i % 6 != 1 and i%6!=2]
#valid_data = [all_data[j] for i, j in enumerate(random_order) if i % 6 == 1]
#test_data = [all_data[j] for i, j in enumerate(random_order) if i % 6 == 2]

#train_data = [all_data[j] for i, j in enumerate(random_order) if i % 5 != 1]
#valid_data = [all_data[j] for i, j in enumerate(random_order) if i % 5 == 1]
#test_data = [all_data[j] for i, j in enumerate(random_order) if i % 6 == 2]



# 建立分词器

#val_test_data = load_data('./data/dev_20200228.csv')

tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=True):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for i in idxs:
            #print(self.data[i])
            _,_, text1, text2, label = self.data[i]
#            text1 = reproduce_text(text1)
#            text2 = reproduce_text(text2)

            token_ids, segment_ids = tokenizer.encode(text1, text2, max_length=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                
#                transfer_flag = np.random.rand()
#                if transfer_flag>0.8:
#                    batch_token_ids, batch_segment_ids = batch_segment_ids, batch_token_ids
                
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []
                                
# 加载预训练模型:: 华为
#bert = build_bert_model(
#    config_path=config_path,
#    checkpoint_path=checkpoint_path,
#    model='nezha',
#    with_pool=True,
#    return_keras_model=False,
#)


## 加载预训练模型:: 哈工大 RBT3
#bert = build_bert_model(
#    config_path=config_path,
#    checkpoint_path=checkpoint_path,
#    with_pool=True,
#    return_keras_model=False,
#)

##加载默认Bert
#bert = build_bert_model(
#    config_path=config_path,
#    checkpoint_path=checkpoint_path,
#    with_pool=True,
#    return_keras_model=False,
#)


##加载哈工大 Robert_ext
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


output = Dropout(rate=0.01)(bert.model.output)
#output_svm = Dense(units=256,
#                   activation='relu',
#                   kernel_initializer=bert.initializer)(output)

output = Dense(units=2,
               activation='softmax',
               kernel_initializer=bert.initializer)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()

##adding gradient panalty
def sparse_categorical_crossentropy(y_true, y_pred):
    """自定义稀疏交叉熵
    这主要是因为tf自带的sparse_categorical_crossentropy不支持求二阶梯度。
    """
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    y_true = K.one_hot(y_true, K.shape(y_pred)[-1])
    return K.mean(K.categorical_crossentropy(y_true, y_pred))


def loss_with_gradient_penalty(y_true, y_pred):
    """带梯度惩罚的loss
    """
    loss = sparse_categorical_crossentropy(y_true, y_pred)
    embeddings = search_layer(y_pred, 'Embedding-Token').embeddings
    gp = K.sum(K.gradients(loss, [embeddings])[0].values**2)
    return loss + 0.5 * gp


model.compile(
    loss=loss_with_gradient_penalty,
    optimizer=Adam(lr),  # 用足够小的学习率
    #optimizer=PiecewiseLinearLearningRate(Adam(5e-5), {10000: 1, 30000: 0.1}),
    metrics=['accuracy'],
)


##网上攻略
#model.compile(
#    loss='binary_crossentropy',
#    # 迷信之RADAM
#    optimizer=RAdam(1e-5),  # 用足够小的学习率
#    # 下面这个就是论文中提到的优秀的论文
#    # optimizer=PiecewiseLinearLearningRate(Adam(1e-5), {1000: 1e-5, 2000: 6e-5}),
#    metrics=['accuracy']


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self, best_val_acc=0.):
        self.best_val_acc = best_val_acc

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(test_generator)
        if val_acc > self.best_val_acc:
            print('Model ckpt store!')
            self.best_val_acc = val_acc
            model.save_weights('{0}_best_model.weights'.format(prefix))
        test_acc = evaluate(valid_generator)
        print(u'test_acc: %.5f, best_test_acc: %.5f, val_acc: %.5f\n'
              % (val_acc, self.best_val_acc, test_acc))
#        print(u'val_acc: %.5f, best_val_acc: %.5f'
#              % (val_acc, self.best_val_acc))

all_data = load_data('./data/train.csv')
random_order = range(len(all_data))
np.random.shuffle(list(random_order))
        
train_data = [all_data[j] for i, j in enumerate(random_order) if i % 6 != 1 and i%6!=2]
valid_data = [all_data[j] for i, j in enumerate(random_order) if i % 6 == 1]
test_data = [all_data[j] for i, j in enumerate(random_order) if i % 6 == 2]
# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)

evaluator = Evaluator()
model.fit_generator(train_generator.forfit(),
                    steps_per_epoch=len(train_generator),
                    epochs=epoch_num,
                    callbacks=[evaluator])

model.load_weights('{0}_best_model.weights'.format(prefix))
best_acc = evaluate(test_generator)

#==================second turn===========================
train_data = [all_data[j] for i, j in enumerate(random_order) if i % 6 != 3 and i%6!=2]
valid_data = [all_data[j] for i, j in enumerate(random_order) if i % 6 == 3]
test_data = [all_data[j] for i, j in enumerate(random_order) if i % 6 == 2]
# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)

evaluator = Evaluator(best_acc)
model.fit_generator(train_generator.forfit(),
                    steps_per_epoch=len(train_generator),
                    epochs=5,
                    callbacks=[evaluator])

model.load_weights('{0}_best_model.weights'.format(prefix))
best_acc = evaluate(test_generator)


#===================third turn===========================
print('******************Three******************')
train_data = [all_data[j] for i, j in enumerate(random_order) if i % 6 != 4 and i%6!=2]
valid_data = [all_data[j] for i, j in enumerate(random_order) if i % 6 == 4]
test_data = [all_data[j] for i, j in enumerate(random_order) if i % 6 == 2]
# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)

evaluator = Evaluator(best_acc)
model.fit_generator(train_generator.forfit(),
                    steps_per_epoch=len(train_generator),
                    epochs=8,
                    callbacks=[evaluator])

model.load_weights('{0}_best_model.weights'.format(prefix))
best_acc = evaluate(test_generator)

##=====================================4===========
print('******************Four******************')
train_data = [all_data[j] for i, j in enumerate(random_order) if i % 6 != 5 and i%6!=2]
valid_data = [all_data[j] for i, j in enumerate(random_order) if i % 6 == 5]
test_data = [all_data[j] for i, j in enumerate(random_order) if i % 6 == 2]
# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)

evaluator = Evaluator(best_acc)
model.fit_generator(train_generator.forfit(),
                    steps_per_epoch=len(train_generator),
                    epochs=5,
                    callbacks=[evaluator])

model.load_weights('{0}_best_model.weights'.format(prefix))
best_acc = evaluate(test_generator)

#==================================5
train_data = [all_data[j] for i, j in enumerate(random_order) if i % 6 != 0 and i%6!=2]
valid_data = [all_data[j] for i, j in enumerate(random_order) if i % 6 == 0]
test_data = [all_data[j] for i, j in enumerate(random_order) if i % 6 == 2]
# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)

evaluator = Evaluator(best_acc)
model.fit_generator(train_generator.forfit(),
                    steps_per_epoch=len(train_generator),
                    epochs=5,
                    callbacks=[evaluator])

model.load_weights('{0}_best_model.weights'.format(prefix))
best_acc = evaluate(test_generator)

