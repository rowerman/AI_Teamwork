# 使用说明
## 安装所需要的文件：
1. 按照 `requirement.txt`安装所需要的库：运行命令`pip install -r requirements.txt`
2. 请依据个人所安装的CUDA版本以及python版本安装相应的以下三个包(如果仅仅做测试请跳过此步骤)，以下为三个包的名字以及教程：
	1. torch-1.13.1+cu117-cp310-cp310-win_amd64.whl
	2. torchaudio-0.13.1+cu117-cp310-cp310-win_amd64.whl
	3. torchvision-0.14.1+cu117-cp310-cp310-win_amd64.whl
	4. 具体教程如下：[参考链接](https://zhuanlan.zhihu.com/p/612181449)
## 运行&训练
1. 如果想要测试请使用经过训练的示例网络：
	python aigcmn.py
	同时 `aigcmn.py`提供了一个类接口 `AiGcMn()` ,直接运行 `aigcmn.py`可以进行测试体验
2. 如果想要从头训练网络请运行 `train.py`：
	python train.py

## 文件作用说明:

- 每250个iters会输出一张结果演示图在`pictue_output`文件夹里
- 每次得到更优网络会保存在 `modelD`和 `modelG`文件夹里
- 模型类文件在 `model.py`里面
- 数据载入文件在 `makeData.py`里面




# 接口文件说明
- 接口类文件aigcmn.py说明：
	- `class：AiGcMn`，类的初始化函数中完成模型加载等初始化工作。
		- 初始化函数`__init__(self)`
			- 函数的参数：
				- `self`：实例对象本身
			- 初始化内容：
				- `self.G=Generator()`，调用model.py文件里的`Class Generator`并将这个对象实例化。
				- `self.G.load_state_dict(torch.load("GAN_ZYN\modelG\model_100_0.pt"))`，加载Generator模型文件。
		- 接口函数`generator(self,Labels)`
			- 函数的参数：
				- `self`：实例对象本身
				- `Labels`：要生成的数字list
			- 函数的功能描述：
				- 接口函数 `generator()` 中生成随机噪音然后将噪音与提供的Labels加在一起，通过 `self.G()` 产生结果。
			- 函数的返回值：所需的 64\*1\*28\*28 的 tensor
	- 实例化说明`function EXAMPLE_PI_20()`：
		- 作用：输出20张π的前64位，并创建文件夹EXAMPLE_PI_OUTPUT，将图片保存其中
		- 输入：`Label=\[3141592653589793238462643383279502884197169399375105820974944592\]`
		- 输出：输出是64\*\1\*28\*\28的tensor，下图为示例输出的其中一张图片
		![[test_pic_4.png]]
	- 个性化定制输入Labels `function EXAMPLE_20()`：
		- 功能：输入要生成的数字(不超过64位的整数),如果直接回车默认输出π，并创建文件夹EXAMPLE_OUTPUT，将图片保存其中
	- **彩蛋！！！**
		- 请缩小视图查看彩蛋

