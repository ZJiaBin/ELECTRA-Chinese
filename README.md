# ELECTRA-Chinese

提供了在哈工大开源的中文模型参数 https://github.com/ymcui/Chinese-ELECTRA  的基础上继续做预训练的方法

主要遇到的问题是哈工大开源的Electra只保留了inference需要的参数，但没有提供比如Adam相关的参数，导致了在尝试加载该模型继续训练时由于参数不全报错
解决方法：

   首先随机生成一个参数完整的ckpt文件
   之后依次加载哈工大开源的模型，以及上述生成的文件，将哈工大模型中出现的参数给生成的新模型重新初始化，其他的参数保持不变
   具体脚本在 generate_chinese_model.py中,脚本中涉及到一个variables.pikle文件，已上传在百度云盘
      
      该文件是从哈工大模型中提取的参数名和参数值(electra-base)
      链接：https://pan.baidu.com/s/13FwCV9ET76LOBLbjTqiBlA 
      提取码：jzrp 

   
   

关于训练的一些说明:
官方提供的脚本非常友好，也已经支持中文了，不需要任何改动可以直接跑, 一些路径相关的参数需要提前指定好

    step1:生成训练数据
        build_pretraining_dataset.py
    step2:预训练
        run_pretraining.py
        需要指定一个空的输出目录，模型开始训练之后即可停止程序，会生成step0时的ckpt文件，接下来需要通过generate_chinese_model.py中的脚本对参数进行初始化
        之后再重新执行run_pretraining.py即可在预训练好的模型基础上继续训练
        
另外之所以尝试在哈工大开源的参数的基础上做预训练是因为本来尝试了从随机值开始训，但是发现模型很难收敛，loss一直在20上下，测试发现discriminator输出一直是同一个标签，还没找到原因
太蠢了。。。是因为用了脚本里默认的学习率，学习率跟论文保持一致就好了。。。。
——————————————————————————

参考:

    https://github.com/google-research/electra
    https://github.com/ymcui/Chinese-ELECTRA
