
#准备
# 下载数据，按结构放置

## **Training**
The data folders should be:
```
Dataset
    * inpainting
        - train_A # 去阴影前的原图 images
        - train_B # 留空文件夹
        - train_C # gts

    *result      # 预测结果图片保存路径
    *test_dataaet # 待测试数据集
```

## **脚本说明**
```
1）infer.py
    去阴影testB采用该脚本生成，存在动态shape。对Dataset/test_dataset中的图片推理结果，保存于Dataset/result文件夹
    python infer.py

2）ckpt2pb.py
    由tf快照生成.pb模型脚本, .pb保存于pd_model中
    python ckpt2pb.py
    
3）x2paddle_code.py
    该脚本是由paddle工具转换.pb模型来的,用于paddle推理，paddle框架推理用。但目前paddle尚不支持动态shape的.pb转换，
	不推荐采用此脚本预测；
    对Dataset/test_dataset中的图片推理结果，保存于Dataset/result文件夹
    python x2paddle_code.py 

4）train.py
    训练脚本。模型保存于logs/pre-trained；同时训练过程中测试效果图亦保存于此文件夹。
    python train.py 

```

This repo base on  the code:
    https://github.com/vinthony/ghost-free-shadow-removal
    与该仓库去除阴影原理类似，因为没有涉及mask预测分支，故模型设计仅取预测target分支即可