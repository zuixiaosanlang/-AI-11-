比赛详情链接 https://aistudio.baidu.com/aistudio/competition/detail/704/0/introduction

整工程下载（包含训练数据集、预训练模型、提交paddle文件）
链接：https://pan.baidu.com/s/1VRfFS29VEhtbemt0VDipNg 
提取码：asdf 
--来自百度网盘超级会员V5的分享


github上的缺少数据集、预训练模型、提交paddle文件

一、整体方案：

	对抗生成网络生成表格线+yolov5目标检测；
	经实验，如果直接使用yolov5进行表格检测，发现效果并不好；加入对抗生成网络生成表格线进行辅助后，yolov5训练检测效果大大提升；
	其中生成表格线的对抗生成网络代码参考https://github.com/vinthony/ghost-free-shadow-removal，仅截取使用其目标图像生成分支，去除生成mask分支；
	目标检测选用yolov5最新发布版本：v7.0 - YOLOv5 SOTA Realtime Instance Segmentation。https://github.com/ultralytics/yolov5/releases/tag/v7.0
	
	
二、复现

	数据集、预训练模型等均包含在工程目录内部,相关目录路径均采用相对路径，已配置好在程序内部，配置好环境执行几个命令即可复现；
	工程下载链接
	
	1. 目录介绍
		gan_line为对抗生成网络代码，预测表格线；
		yolov5-7.0为yolov5目标检测代码，检测3类别表格：行、列、合并，大表格可由这些表格一起推断出（最大外包围框记为大表格）
		submit_files为提交paddle文件
		
		gan_line 中的tensorflow采用tf1.x版本
		
	
	2. 数据预处理；
	对抗生成网络训练数据集：防止靠近图像边缘的表格生成不好，在原图4周补32厚度的0像素；
	yolov5数据集：抗生成网络预测的图片（480*480），令表格线像素变为红色，保存为yolov5训练数据集；
	数据集均已处理好放于工程内部
	
	3.抗生成网络模型
	1）训练
	   cd gan_line
	   python train.py
	2）快照转.pb文件
	   python ckpt2pb.py
	3) .pb转paddle模型
	   cd pd_model
	   x2paddle --framework=tensorflow --model=frozen_model.pb --save_dir=pd_model
	   pd_model文件夹中的inference_model推理文件模型即为提交模型文件submit_files/pd_model/inference_model
	   
	4. 目标检测
	1）训练
		cd yolov5-7.0
		python train.py
		成绩中模型选用yolov5x.pt模型进行迁移学习；训练生成遵从yolov5，相关数据及模型保存在runs目录.
		
	2) yolo模型转为paddle模型
	   python export.py
	   会生成 best-seg_paddle_model目录文件,即为paddle模型文件，拷贝至submit_files文件夹覆盖即可.
	
	5.进入目录打包所有文件提交
	   cd submit_files
	   zip -r submit_yolov5.zip ./
	
三、分数提升探讨
	
	1.yolo分割模型中的输出并不是直接的mask，后面的运算过程均基于pytorch环境进行的，这样的代码提取出来后不能直接提交否则会报错。为解决这个问题，本人将其改成基于numpy环境的提交；如果能将该部分直接放到模型内部，应该会更好；
	
	2.yolov5检测出的目标框，新增步骤结合对抗生成网络生成的表格线，修正目标框，分数应该会更高；
	
	3.关于yolov5的调优还是有很多的，可以修改默认的超参数文件data\hyps\hyp.scratch-low.yaml
	
	4.里面关于目标检测的非极大值抑制算法non_max_suppression_np(),特别是conf_thres参数，数值过大，会漏表格目标，过小会把非表格图像检测出来，影响P,PR分数。

