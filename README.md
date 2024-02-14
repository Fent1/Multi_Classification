# 多分类任务
中文 | [英文](./README_EN.md)
## 课题背景
Pychex公司想更详细地了解他们的目标客户群体，向我们提供了一份关于10000+公司的网站地址以及他们的公司类型。Pychex希望我们能构建一个模型能够通过提供该公司网站后准确分类得到该公司所属类型。


## 数据集
10000+行公司网站数据集中，共有17种公司类型，如下图所示：

<img src="https://github.com/Fent1/Multi_Classification/assets/43925272/09261a1c-b16d-4845-8eee-cfe781f39eee" alt="image" width="500" height="auto">


## 网络爬虫
由于提供的数据集含有网站链接，我们需要用到网络爬虫提取网站中可用的信息，并新生成为Text列，作为自变量用于训练模型。
由于10000+网站爬虫耗时巨大，为了加快爬虫速度，我们采用多进程的方法，利用多个cpu进行计算和信息读取，将爬虫完成时间缩减到6个小时。

<img src="https://github.com/Fent1/Multi_Classification/assets/43925272/2f879f23-714b-47b0-a3b5-46c90ec570ff" alt="image" width="300" height="auto">

提取网站中的句子作为有用的信息

## 模型选择
作为多分类问题的模型选择上，BERT和ERNIE都是比较好用的成熟框架。两个模型在训练后的精确度上没有很大差别，所以我们这里选择使用ERNIE模型进行训练。

## 模型表现

<img src="https://github.com/Fent1/Multi_Classification/assets/43925272/b22b5893-5585-4d87-a52f-f127c06507dc" alt="image" width="300" height="auto">

 - 对于这个数据集，模型性能很好，与之前的模型相比，准确率从30.42%增加到55.60%。
 - 精确率和召回率都很低，分别为32.37和35.51，这意味着这个分类模型仍然丢失一些特征的获取。
 - 由于每个标签可能存在严重的不平衡，我们的模型性能高但F1分数低。

## 如何运行

 1. 请先运行网络爬虫代码，请直接在Business_Industry_URLS.csv同目录下运行Starter_Web_Scraper_2.py。
 2. 运行完成后会生Business_Industry_URLS_wText.csv文件，请在main.py中修改train_path对应的文件地址。
 3. 运行main.py。

P.S: 由于某些原因现在还没有上传数据集Business_Industry_URLS.csv
