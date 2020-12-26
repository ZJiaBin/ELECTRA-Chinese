# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pre-trains an ELECTRA model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
import collections
import json
import tensorflow.compat.v1 as tf

def model_init():
  """
  Chinese-electra提供的参数，adam相关的的参数并没有提供，所以没有办法直接拿过来在它的基础上预训练（restore的时候会报错）
  解决方案：
    首先需要先执行一下预训练脚本 run_pretraining.py, 开始迭代出现loss的时候停掉即可，这时候指定输出目录中会出现一版ckpt文件，
    此时的参数是全的，electra-base的大小大概1.2g左右，但参数值都是随机初始化的形式，需要加载之后手动初始化一下
    需要先加载哈工大提供的模型，获取变量名，和变量值，在通过 tf.assign()的形式给参数赋值，再重新存储即可

    tf.train.list_variables(model_path) 可以获取ckpt中的变量名，返回的是list of  (name, shape)
    tf.train.load_variable(model, param_name) 可以加载变量名的值
  """
  import pickle
  from tqdm import tqdm

  # 加载提取完的变量字典
  with open('chinese_model_variables/variables.pikle', 'rb') as fin:
      name2variable = pickle.load(fin)

  with tf.Session() as sess:
    saver = tf.train.import_meta_graph('your_model_path/model.ckpt-0.meta')  # 加载模型结构
    saver.restore(sess= sess, save_path= 'your_model_path/model.ckpt-0')  # 只需要指定目录就可以恢复所有变量信息
    name_list = [name + ':0' for name in name2variable.keys()]
    for v in tqdm(tf.all_variables()):
        n = v.name
        if 'global_step' in n: continue
        if n in name_list:
            print('assign variable {}'.format(n))
            sess.run(tf.assign(v, name2variable.get(n.replace(':0',''))))
            # res = sess.run(v)
            pass
    saver = tf.train.Saver()
    saver.save(sess, save_path= 'your_model_path/model.ckpt-0' )
def main():
  model_init()


if __name__ == "__main__":
  main()
