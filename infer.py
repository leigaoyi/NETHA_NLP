# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 19:27:50 2020

@author: kasy
"""

from keras.layers import *

from bert4keras.backend import keras, set_gelu
from bert4keras.bert import build_bert_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.tokenizer import Tokenizer
import pandas as pd
import numpy as np


#set_gelu('tanh')  # 切换gelu版本

epoch_num = 15
prefix = 'NEZHA'
maxlen = 128

def load_data(filename):
    D = pd.read_csv(filename).values.tolist()
    return D

import re

ch_reg = "[\u002c\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\u300a\u300b]"

f = lambda x:re.sub(ch_reg, '，', x).strip("?").strip('？').replace(ch_reg, '，').strip('，')

#def load_data(filename):
#    D = pd.read_csv(filename)
#    D['query1'] = D['query1'].map(f)
#    D['query2'] = D['query2'].map(f)
#    D = D.values.tolist()
#    return D


#test_data = load_data('./data/dev_20200228.csv')

test_data = load_data('/tcdata/test.csv')


batch_size = 1


# HuaWei NeTha
config_path = 'NEZHA/bert_config.json'
checkpoint_path = 'NEZHA/model.ckpt-900000'
dict_path = 'NEZHA/vocab.txt'

# 加载数据集
#all_data = load_data('./data/train_20200228.csv')
#random_order = range(len(all_data))
#np.random.shuffle(list(random_order))
#train_data = [all_data[j] for i, j in enumerate(random_order) if i % 6 != 1 and i%6!=2]
#valid_data = [all_data[j] for i, j in enumerate(random_order) if i % 6 == 1]
#test_data = [all_data[j] for i, j in enumerate(random_order) if i % 6 == 2]
# 建立分词器
    


tokenizer = Tokenizer(dict_path, do_lower_case=True)


class test_data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for i in idxs:
            #print(self.data[i])
            idx_i,_, text1, text2, label = self.data[i]
#             print(text1, text2, label)
            token_ids, segment_ids = tokenizer.encode(text1, text2, max_length=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            #batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                #batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], idx_i
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []
                
                
##加载预训练模型:: 华为
bert = build_bert_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='nezha',
    with_pool=True,
    return_keras_model=False,
)    

#if prefix=='HUAWEI':
#     #加载预训练模型:: 华为
#    bert = build_bert_model(
#        config_path=config_path,
#        checkpoint_path=checkpoint_path,
#        model='nezha',
#        with_pool=True,
#        return_keras_model=False,
#    )


output = Dropout(rate=0.04)(bert.model.output)
output = Dense(units=2,
               activation='softmax',
               kernel_initializer=bert.initializer)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()


def eval_submission(data):
    #total, right = 0., 0.
    pred_list = []
    idx_list = []
    for x_true, y_idx in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        #y_true = y_true[:, 0]
        pred_list.append(int(y_pred))
        idx_list.append(y_idx)
        #print(y_idx, y_pred)
    pred_list = np.asarray(pred_list)
    return idx_list, pred_list

test_generator = test_data_generator(test_data, 1)
print('{0}_best_model.weights'.format(prefix))


model.load_weights('{0}_best_1_model.weights'.format(prefix))
idxs1, preds1 = eval_submission(test_generator)

print('Beging first')
model.load_weights('{0}_best_2_model.weights'.format(prefix))
idxs1, preds2 = eval_submission(test_generator)

print('Begin three')
model.load_weights('{0}_best_3_model.weights'.format(prefix))
idxs1, preds3 = eval_submission(test_generator)

model.load_weights('{0}_best_4_model.weights'.format(prefix))
idxs1, preds4 = eval_submission(test_generator)

print('Begin last')
model.load_weights('{0}_best_5_model.weights'.format(prefix))
idxs1, preds5 = eval_submission(test_generator)

preds = preds1 + preds2 + preds3 + preds4 + preds5
preds_num = len(preds)

preds_final = []
for i in range(preds_num):
    if preds[i]>2.5:
        preds_final.append(1)
    else:
        preds_final.append(0)
#preds[preds>2.5] = 1
#preds[preds<=2.5] = 0

submission = pd.DataFrame({'id': idxs1,
                         'label': preds_final})

submission_file = 'result.csv'
submission.to_csv(submission_file, index=False)


