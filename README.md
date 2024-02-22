# 多分类任务
中文 | [英文](./README_EN.md)
## 课题背景
Pychex公司想更详细地了解他们的目标客户群体，向我们提供了一份关于10000+公司的网站地址以及他们的公司类型。Pychex希望我们能构建一个模型能够通过提供该公司网站后准确分类得到该公司所属类型。
本次课题的思路是通过爬虫软件爬取每个公司网站中含有关键信息的文本，然后用NLP模型训练文本完成多分类任务。
## 数据集
10000+行公司网站数据集中，共有138种公司类型，即分类任务会将文本分为138个类，类别数量分布如下图所示：

<img src="https://github.com/Fent1/Multi_Classification/assets/43925272/88967fb3-1ba7-4e55-bfce-6f9317117fdc" alt="image" width="300" height="auto">

## 网络爬虫
由于提供的数据集含有网站链接，我们需要用到网络爬虫提取网站中可用的信息，并新生成为Text列，作为自变量用于训练模型。
由于10000+网站爬虫耗时巨大，为了加快爬虫速度，我们采用多进程的方法，利用多个cpu进行计算和信息读取，将爬虫完成时间缩减到6个小时。

<img src="https://github.com/Fent1/Multi_Classification/assets/43925272/2f879f23-714b-47b0-a3b5-46c90ec570ff" alt="image" width="300" height="auto">

提取网站中的句子作为有用的信息

## 模型选择
本模型是基于ernie-2.0-base-en在40多万条英文商品标题数据集-AE数据集上微调的。该模型可以针对英文的文本进行快速的无监督分类。
ERNIE2.0 构建了三种类型的无监督任务。为了完成多任务的训练，又提出了连续多任务学习，整体框架见下图。

![image](https://github.com/Fent1/Multi_Classification/assets/43925272/c5db9b53-e038-4086-ad18-6881d7dc4de8)

Erine-2.0提供三种训练策略：

![image](https://github.com/Fent1/Multi_Classification/assets/43925272/96fbccf6-5c12-4b20-a081-febd4720b8ce)

策略一：Multi-task Learning，就是让模型同时学这3个任务，具体的让这3个任务的损失函数权重相加，然后一起反向传播更新参数；

策略二：Continual Learning，先训练任务1，再训练任务2，再训练任务3，这种策略的缺点是容易遗忘前面任务的训练结果，最后训练出的模型容易对最后一个任务过拟合；

策略三：Sequential Multi-task Learning，连续多任务学习，即第一轮的时候，先训练任务1，但不完全让它收敛训练完，第二轮，一起训练任务1和任务2，同样不让模型收敛完，第三轮，一起训练三个任务，直到模型收敛完。

## 微调数据介绍
train.txt: 训练集，共有137个类别,5841行数据
val.txt：验证集，共有137个类别，1000行数据

## 数据预处理
文本与标签类别名用tab符'^'分隔开，标签用英文逗号','分隔开。

train.txt/val.txt/test.txt 文件格式：

    <文本>^<标签>
    <文本>^<标签>

## 数据增强
为了使数据中的标签平衡增加模型表现，我们采用过采样(Oversampling)方法：
通过数据增强方法，如同义词替换等增加数据量少的标签的数据量。

                !python PaddleNLP/applications/text_classification/multi_class/analysis/aug.py \
    --create_n 2 \
    --aug_percent 0.1 \
    --aug_strategy mix \
    --aug_type mlm \
    --train_path "data/train.txt" \
    --aug_path "data/aug.txt"

合并数据增强和原训练集的数据：

                # Combine Augmented data and training data together
                !cat data/aug.txt data/train.txt > data/train_aug.txt
                
## 模型表现
输出打印示例：

                00:37:33,787] [    INFO] - -----Evaluate model-------
                [2023-12-08 00:37:33,787] [    INFO] - Dev dataset size: 1000
                [2023-12-08 00:37:33,787] [    INFO] - Accuracy in dev dataset: 53.60%
                [2023-12-08 00:37:33,787] [    INFO] - Macro average | precision: 27.94 | recall: 29.73 | F1 score 27.99
                [2023-12-08 00:37:33,787] [    INFO] - Class name: Personal Care Services
                [2023-12-08 00:37:33,787] [    INFO] - Evaluation examples in dev dataset: 28(2.8%) | precision: 73.08 | recall: 67.86 | F1 score 70.37
                [2023-12-08 00:37:33,787] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,787] [    INFO] - Class name: Other Ambulatory Health Care Services
                [2023-12-08 00:37:33,787] [    INFO] - Evaluation examples in dev dataset: 25(2.5%) | precision: 44.00 | recall: 44.00 | F1 score 44.00
                [2023-12-08 00:37:33,788] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,788] [    INFO] - Class name: Beer  Wine  and Distilled Alcoholic Beverage Merchant Wholesalers
                [2023-12-08 00:37:33,788] [    INFO] - Evaluation examples in dev dataset: 0 (0%)
                [2023-12-08 00:37:33,788] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,788] [    INFO] - Class name: Management  Scientific  and Technical Consulting Services


<img src="https://github.com/Fent1/Multi_Classification/assets/43925272/b22b5893-5585-4d87-a52f-f127c06507dc" alt="image" width="300" height="auto">

 - 对于这个数据集，模型性能很好，与之前的模型相比，准确率从30.42%增加到55.60%。
 - 精确率和召回率都很低，分别为32.37和35.51，这意味着这个分类模型仍然丢失一些特征的获取。
 - 由于每个标签可能存在严重的不平衡，我们的模型性能高但F1分数低。

## 模型预测

                from paddlenlp import Taskflow

                cls = Taskflow("text_classification", task_path='checkpoint/export', is_static_model=True)
                cls(test_df.iloc[0, 2])
                                [2023-12-08 00:37:37,492] [    INFO] - Load id2label from checkpoint/export/id2label.json.
                                [{'predictions': [{'label': 'Personal Care Services',
                                    'score': 0.9765003992179774}],
                                  'text': 'Mariposa Hair Salon:When you choose Mariposa at Jason Avenue, youre choosing a life of style and sophistication. Youll enjoy top-of-the-line in-home features to make your everyday routine easy, from high-speed internet and cable TV access to a generous porch or balcony with a storage closet. Once you step outside, youll be greeted by a host of community amenities perfect for active 55+ adults, from a dog park to a petanque court and horseshoe pit. Make every minute count at Mariposa at Jason Avenue. The on'}]

## 如何运行

 1. 请先运行网络爬虫代码，请直接在Business_Industry_URLS.csv同目录下运行Starter_Web_Scraper_2.py。
 2. 运行完成后会生Business_Industry_URLS_wText.csv文件，请在main.py中修改train_path对应的文件地址。
 3. 运行main.py。

P.S: 由于某些原因现在还没有上传数据集Business_Industry_URLS.csv
