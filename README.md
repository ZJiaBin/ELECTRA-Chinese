# ELECTRA-Chinese
# electra_Chinese 训练脚本

提供了在哈工大开源的中文模型参数（https://github.com/ymcui/Chinese-ELECTRA）的基础上继续做预训练的方法

主要遇到的问题是哈工大开源的Electra只保留了inference需要的参数，但没有提供比如Adam相关的参数，导致了在尝试加载该模型继续训练时由于参数不全报错
解决方法：
   首先随机生成一个参数完整的ckpt文件
   之后依次加载哈工大开源的模型，以及上述生成的文件，将哈工大模型中出现的参数给生成的新模型重新初始化，其他的参数保持不变
   具体脚本在 generate_chinese_model.py中


参考:
    https://github.com/google-research/electra
    https://github.com/ymcui/Chinese-ELECTRA
