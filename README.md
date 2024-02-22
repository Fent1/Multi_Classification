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

## 模型表现
输出打印示例：

                    [2023-12-08 00:37:25,240] [ WARNING] - evaluation_strategy reset to IntervalStrategy.STEPS for do_eval is True. you can also set evaluation_strategy='epoch'.
                [2023-12-08 00:37:25,240] [    INFO] - using `logging_steps` to initialize `eval_steps` to 100
                [2023-12-08 00:37:25,240] [    INFO] - The default value for the training argument `--report_to` will change in v5 (from all installed integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as now. You should start updating your code and make this info disappear :-).
                [2023-12-08 00:37:25,242] [    INFO] - We are using <class 'paddlenlp.transformers.ernie.modeling.ErnieForSequenceClassification'> to load 'checkpoint'.
                [2023-12-08 00:37:25,242] [    INFO] - loading configuration file checkpoint/config.json
                [2023-12-08 00:37:25,244] [    INFO] - Model config ErnieConfig {
                  "architectures": [
                    "ErnieForSequenceClassification"
                  ],
                  "attention_probs_dropout_prob": 0.1,
                  "dtype": "float32",
                  "enable_recompute": false,
                  "fuse": false,
                  "hidden_act": "gelu",
                  "hidden_dropout_prob": 0.1,
                  "hidden_size": 768,
                  "id2label": {
                    "0": "Personal Care Services",
                    "1": "Other Ambulatory Health Care Services",
                    "2": "Beer  Wine  and Distilled Alcoholic Beverage Merchant Wholesalers",
                    "3": "Management  Scientific  and Technical Consulting Services",
                    "4": "Other Amusement and Recreation Industries",
                    "5": "Architectural  Engineering  and Related Services",
                    "6": "Business  Professional  Labor  Political  and Similar Organizations",
                    "7": "Offices of Other Health Practitioners",
                    "8": "Motion Picture and Video Industries",
                    "9": "Residential Building Construction",
                    "10": "Civic and Social Organizations",
                    "11": "Services to Buildings and Dwellings",
                    "12": "Scientific Research and Development Services",
                    "13": "Legal Services",
                    "14": "Other Schools and Instruction",
                    "15": "Personal and Household Goods Repair and Maintenance",
                    "16": "Traveler Accommodation",
                    "17": "Restaurants and Other Eating Places",
                    "18": "Agencies  Brokerages  and Other Insurance Related Activities",
                    "19": "Elementary and Secondary Schools",
                    "20": "Outpatient Care Centers",
                    "21": "Offices of Real Estate Agents and Brokers",
                    "22": "Management of Companies and Enterprises",
                    "23": "Other Specialty Trade Contractors",
                    "24": "Individual and Family Services",
                    "25": "Office Administrative Services",
                    "26": "Employment Services",
                    "27": "Social Advocacy Organizations",
                    "28": "Offices of Dentists",
                    "29": "Lessors of Real Estate",
                    "30": "Religious Organizations",
                    "31": "Land Subdivision",
                    "32": "Offices of Physicians",
                    "33": "Independent Artists  Writers  and Performers",
                    "34": "Administration of Housing Programs  Urban Planning  and Community Development",
                    "35": "Other Financial Investment Activities",
                    "36": "Electronic and Precision Equipment Repair and Maintenance",
                    "37": "Activities Related to Real Estate",
                    "38": "Other Personal Services",
                    "39": "Other Professional  Scientific  and Technical Services",
                    "40": "Justice  Public Order  and Safety Activities",
                    "41": "Computer Systems Design and Related Services",
                    "42": "Building Equipment Contractors",
                    "43": "Other Heavy and Civil Engineering Construction",
                    "44": "Miscellaneous Durable Goods Merchant Wholesalers",
                    "45": "Community Food and Housing  and Emergency and Other Relief Services",
                    "46": "Hardware  and Plumbing and Heating Equipment and Supplies Merchant Wholesalers",
                    "47": "Professional and Commercial Equipment and Supplies Merchant Wholesalers",
                    "48": "Commercial and Industrial Machinery and Equipment (except Automotive and Electronic) Repair and Maintenance",
                    "49": "Farm Product Raw Material Merchant Wholesalers",
                    "50": "Automotive Repair and Maintenance",
                    "51": "Machinery  Equipment  and Supplies Merchant Wholesalers",
                    "52": "Drinking Places (Alcoholic Beverages)",
                    "53": "Grantmaking and Giving Services",
                    "54": "Nondepository Credit Intermediation",
                    "55": "Child Day Care Services",
                    "56": "Commercial and Industrial Machinery and Equipment Rental and Leasing",
                    "57": "Household Appliances and Electrical and Electronic Goods Merchant Wholesalers",
                    "58": "Travel Arrangement and Reservation Services",
                    "59": "Nursing Care Facilities (Skilled Nursing Facilities)",
                    "60": "Petroleum and Petroleum Products Merchant Wholesalers",
                    "61": "Promoters of Performing Arts  Sports  and Similar Events",
                    "62": "General Medical and Surgical Hospitals",
                    "63": "Accounting  Tax Preparation  Bookkeeping  and Payroll Services",
                    "64": "Foundation  Structure  and Building Exterior Contractors",
                    "65": "Support Activities for Mining",
                    "66": "Nonresidential Building Construction",
                    "67": "Specialized Design Services",
                    "68": "Greenhouse  Nursery  and Floriculture Production",
                    "69": "RV (Recreational Vehicle) Parks and Recreational Camps",
                    "70": "Oilseed and Grain Farming",
                    "71": "Miscellaneous Nondurable Goods Merchant Wholesalers",
                    "72": "Lumber and Other Construction Materials Merchant Wholesalers",
                    "73": "Water  Sewage and Other Systems",
                    "74": "Other Investment Pools and Funds",
                    "75": "Vocational Rehabilitation Services",
                    "76": "Grocery and Related Product Merchant Wholesalers",
                    "77": "Waste Collection",
                    "78": "Museums  Historical Sites  and Similar Institutions",
                    "79": "Consumer Goods Rental",
                    "80": "Activities Related to Credit Intermediation",
                    "81": "Junior Colleges",
                    "82": "Performing Arts Companies",
                    "83": "Other Animal Production",
                    "84": "Executive  Legislative  and Other General Government Support",
                    "85": "Other Support Services",
                    "86": "Spectator Sports",
                    "87": "Residential Intellectual and Developmental Disability  Mental Health  and Substance Abuse Facilities",
                    "88": "Building Finishing Contractors",
                    "89": "Investigation and Security Services",
                    "90": "Technical and Trade Schools",
                    "91": "Home Health Care Services",
                    "92": "Business Support Services",
                    "93": "Other Crop Farming",
                    "94": "Waste Treatment and Disposal",
                    "95": "Highway  Street  and Bridge Construction",
                    "96": "Depository Credit Intermediation",
                    "97": "Advertising  Public Relations  and Related Services",
                    "98": "Paper and Paper Product Merchant Wholesalers",
                    "99": "Motor Vehicle and Motor Vehicle Parts and Supplies Merchant Wholesalers",
                    "100": "Other Residential Care Facilities",
                    "101": "Furniture and Home Furnishing Merchant Wholesalers",
                    "102": "Educational Support Services",
                    "103": "Vegetable and Melon Farming",
                    "104": "Utility System Construction",
                    "105": "Metal and Mineral (except Petroleum) Merchant Wholesalers",
                    "106": "Death Care Services",
                    "107": "Apparel  Piece Goods  and Notions Merchant Wholesalers",
                    "108": "Administration of Environmental Quality Programs",
                    "109": "Electric Power Generation  Transmission and Distribution",
                    "110": "Support Activities for Animal Production",
                    "111": "Colleges  Universities  and Professional Schools",
                    "112": "Agents and Managers for Artists  Athletes  Entertainers  and Other Public Figures",
                    "113": "Insurance Carriers",
                    "114": "Business Schools and Computer and Management Training",
                    "115": "Automotive Equipment Rental and Leasing",
                    "116": "Medical and Diagnostic Laboratories",
                    "117": "Remediation and Other Waste Management Services",
                    "118": "Special Food Services",
                    "119": "Continuing Care Retirement Communities and Assisted Living Facilities for the Elderly",
                    "120": "Data Processing  Hosting  and Related Services",
                    "121": "Administration of Economic Programs",
                    "122": "Drycleaning and Laundry Services",
                    "123": "Support Activities for Crop Production",
                    "124": "Drugs and Druggists' Sundries Merchant Wholesalers",
                    "125": "Chemical and Allied Products Merchant Wholesalers",
                    "126": "Fruit and Tree Nut Farming",
                    "127": "Poultry and Egg Production",
                    "128": "Psychiatric and Substance Abuse Hospitals",
                    "129": "Specialty (except Psychiatric and Substance Abuse) Hospitals",
                    "130": "Nonmetallic Mineral Mining and Quarrying",
                    "131": "Sound Recording Industries",
                    "132": "Administration of Human Resource Programs",
                    "133": "Cattle Ranching and Farming",
                    "134": "Wholesale Electronic Markets and Agents and Brokers",
                    "135": "Natural Gas Distribution",
                    "136": "Amusement Parks and Arcades",
                    "137": "Securities and Commodity Exchanges"
                  },
                  "initializer_range": 0.02,
                  "intermediate_size": 3072,
                  "label2id": {
                    "Accounting  Tax Preparation  Bookkeeping  and Payroll Services": 63,
                    "Activities Related to Credit Intermediation": 80,
                    "Activities Related to Real Estate": 37,
                    "Administration of Economic Programs": 121,
                    "Administration of Environmental Quality Programs": 108,
                    "Administration of Housing Programs  Urban Planning  and Community Development": 34,
                    "Administration of Human Resource Programs": 132,
                    "Advertising  Public Relations  and Related Services": 97,
                    "Agencies  Brokerages  and Other Insurance Related Activities": 18,
                    "Agents and Managers for Artists  Athletes  Entertainers  and Other Public Figures": 112,
                    "Amusement Parks and Arcades": 136,
                    "Apparel  Piece Goods  and Notions Merchant Wholesalers": 107,
                    "Architectural  Engineering  and Related Services": 5,
                    "Automotive Equipment Rental and Leasing": 115,
                    "Automotive Repair and Maintenance": 50,
                    "Beer  Wine  and Distilled Alcoholic Beverage Merchant Wholesalers": 2,
                    "Building Equipment Contractors": 42,
                    "Building Finishing Contractors": 88,
                    "Business  Professional  Labor  Political  and Similar Organizations": 6,
                    "Business Schools and Computer and Management Training": 114,
                    "Business Support Services": 92,
                    "Cattle Ranching and Farming": 133,
                    "Chemical and Allied Products Merchant Wholesalers": 125,
                    "Child Day Care Services": 55,
                    "Civic and Social Organizations": 10,
                    "Colleges  Universities  and Professional Schools": 111,
                    "Commercial and Industrial Machinery and Equipment (except Automotive and Electronic) Repair and Maintenance": 48,
                    "Commercial and Industrial Machinery and Equipment Rental and Leasing": 56,
                    "Community Food and Housing  and Emergency and Other Relief Services": 45,
                    "Computer Systems Design and Related Services": 41,
                    "Consumer Goods Rental": 79,
                    "Continuing Care Retirement Communities and Assisted Living Facilities for the Elderly": 119,
                    "Data Processing  Hosting  and Related Services": 120,
                    "Death Care Services": 106,
                    "Depository Credit Intermediation": 96,
                    "Drinking Places (Alcoholic Beverages)": 52,
                    "Drugs and Druggists' Sundries Merchant Wholesalers": 124,
                    "Drycleaning and Laundry Services": 122,
                    "Educational Support Services": 102,
                    "Electric Power Generation  Transmission and Distribution": 109,
                    "Electronic and Precision Equipment Repair and Maintenance": 36,
                    "Elementary and Secondary Schools": 19,
                    "Employment Services": 26,
                    "Executive  Legislative  and Other General Government Support": 84,
                    "Farm Product Raw Material Merchant Wholesalers": 49,
                    "Foundation  Structure  and Building Exterior Contractors": 64,
                    "Fruit and Tree Nut Farming": 126,
                    "Furniture and Home Furnishing Merchant Wholesalers": 101,
                    "General Medical and Surgical Hospitals": 62,
                    "Grantmaking and Giving Services": 53,
                    "Greenhouse  Nursery  and Floriculture Production": 68,
                    "Grocery and Related Product Merchant Wholesalers": 76,
                    "Hardware  and Plumbing and Heating Equipment and Supplies Merchant Wholesalers": 46,
                    "Highway  Street  and Bridge Construction": 95,
                    "Home Health Care Services": 91,
                    "Household Appliances and Electrical and Electronic Goods Merchant Wholesalers": 57,
                    "Independent Artists  Writers  and Performers": 33,
                    "Individual and Family Services": 24,
                    "Insurance Carriers": 113,
                    "Investigation and Security Services": 89,
                    "Junior Colleges": 81,
                    "Justice  Public Order  and Safety Activities": 40,
                    "Land Subdivision": 31,
                    "Legal Services": 13,
                    "Lessors of Real Estate": 29,
                    "Lumber and Other Construction Materials Merchant Wholesalers": 72,
                    "Machinery  Equipment  and Supplies Merchant Wholesalers": 51,
                    "Management  Scientific  and Technical Consulting Services": 3,
                    "Management of Companies and Enterprises": 22,
                    "Medical and Diagnostic Laboratories": 116,
                    "Metal and Mineral (except Petroleum) Merchant Wholesalers": 105,
                    "Miscellaneous Durable Goods Merchant Wholesalers": 44,
                    "Miscellaneous Nondurable Goods Merchant Wholesalers": 71,
                    "Motion Picture and Video Industries": 8,
                    "Motor Vehicle and Motor Vehicle Parts and Supplies Merchant Wholesalers": 99,
                    "Museums  Historical Sites  and Similar Institutions": 78,
                    "Natural Gas Distribution": 135,
                    "Nondepository Credit Intermediation": 54,
                    "Nonmetallic Mineral Mining and Quarrying": 130,
                    "Nonresidential Building Construction": 66,
                    "Nursing Care Facilities (Skilled Nursing Facilities)": 59,
                    "Office Administrative Services": 25,
                    "Offices of Dentists": 28,
                    "Offices of Other Health Practitioners": 7,
                    "Offices of Physicians": 32,
                    "Offices of Real Estate Agents and Brokers": 21,
                    "Oilseed and Grain Farming": 70,
                    "Other Ambulatory Health Care Services": 1,
                    "Other Amusement and Recreation Industries": 4,
                    "Other Animal Production": 83,
                    "Other Crop Farming": 93,
                    "Other Financial Investment Activities": 35,
                    "Other Heavy and Civil Engineering Construction": 43,
                    "Other Investment Pools and Funds": 74,
                    "Other Personal Services": 38,
                    "Other Professional  Scientific  and Technical Services": 39,
                    "Other Residential Care Facilities": 100,
                    "Other Schools and Instruction": 14,
                    "Other Specialty Trade Contractors": 23,
                    "Other Support Services": 85,
                    "Outpatient Care Centers": 20,
                    "Paper and Paper Product Merchant Wholesalers": 98,
                    "Performing Arts Companies": 82,
                    "Personal Care Services": 0,
                    "Personal and Household Goods Repair and Maintenance": 15,
                    "Petroleum and Petroleum Products Merchant Wholesalers": 60,
                    "Poultry and Egg Production": 127,
                    "Professional and Commercial Equipment and Supplies Merchant Wholesalers": 47,
                    "Promoters of Performing Arts  Sports  and Similar Events": 61,
                    "Psychiatric and Substance Abuse Hospitals": 128,
                    "RV (Recreational Vehicle) Parks and Recreational Camps": 69,
                    "Religious Organizations": 30,
                    "Remediation and Other Waste Management Services": 117,
                    "Residential Building Construction": 9,
                    "Residential Intellectual and Developmental Disability  Mental Health  and Substance Abuse Facilities": 87,
                    "Restaurants and Other Eating Places": 17,
                    "Scientific Research and Development Services": 12,
                    "Securities and Commodity Exchanges": 137,
                    "Services to Buildings and Dwellings": 11,
                    "Social Advocacy Organizations": 27,
                    "Sound Recording Industries": 131,
                    "Special Food Services": 118,
                    "Specialized Design Services": 67,
                    "Specialty (except Psychiatric and Substance Abuse) Hospitals": 129,
                    "Spectator Sports": 86,
                    "Support Activities for Animal Production": 110,
                    "Support Activities for Crop Production": 123,
                    "Support Activities for Mining": 65,
                    "Technical and Trade Schools": 90,
                    "Travel Arrangement and Reservation Services": 58,
                    "Traveler Accommodation": 16,
                    "Utility System Construction": 104,
                    "Vegetable and Melon Farming": 103,
                    "Vocational Rehabilitation Services": 75,
                    "Waste Collection": 77,
                    "Waste Treatment and Disposal": 94,
                    "Water  Sewage and Other Systems": 73,
                    "Wholesale Electronic Markets and Agents and Brokers": 134
                  },
                  "layer_norm_eps": 1e-12,
                  "max_position_embeddings": 512,
                  "model_type": "ernie",
                  "num_attention_heads": 12,
                  "num_hidden_layers": 12,
                  "pad_token_id": 0,
                  "paddlenlp_version": null,
                  "pool_act": "tanh",
                  "task_id": 0,
                  "task_type_vocab_size": 3,
                  "type_vocab_size": 4,
                  "use_task_id": true,
                  "vocab_size": 30522
                }
                
                W1208 00:37:27.364761  6816 gpu_resources.cc:61] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 12.0, Runtime API Version: 11.2
                W1208 00:37:27.368619  6816 gpu_resources.cc:91] device: 0, cuDNN Version: 8.2.
                [2023-12-08 00:37:28,250] [    INFO] - All model checkpoint weights were used when initializing ErnieForSequenceClassification.
                
                [2023-12-08 00:37:28,251] [    INFO] - All the weights of ErnieForSequenceClassification were initialized from the model checkpoint at checkpoint.
                If your task is similar to the task the model of the checkpoint was trained on, you can already use ErnieForSequenceClassification for predictions without further training.
                [2023-12-08 00:37:28,252] [    INFO] - We are using <class 'paddlenlp.transformers.ernie.tokenizer.ErnieTokenizer'> to load 'checkpoint'.
                [2023-12-08 00:37:28,506] [    INFO] - ============================================================
                [2023-12-08 00:37:28,506] [    INFO] -     Training Configuration Arguments    
                [2023-12-08 00:37:28,507] [    INFO] - paddle commit id              :4596b9a22540fb0ea5d369c3c804544de61d03d0
                [2023-12-08 00:37:28,507] [    INFO] - _no_sync_in_gradient_accumulation:True
                [2023-12-08 00:37:28,507] [    INFO] - activation_quantize_type      :None
                [2023-12-08 00:37:28,507] [    INFO] - adam_beta1                    :0.9
                [2023-12-08 00:37:28,507] [    INFO] - adam_beta2                    :0.999
                [2023-12-08 00:37:28,507] [    INFO] - adam_epsilon                  :1e-08
                [2023-12-08 00:37:28,507] [    INFO] - algo_list                     :None
                [2023-12-08 00:37:28,507] [    INFO] - batch_num_list                :None
                [2023-12-08 00:37:28,507] [    INFO] - batch_size_list               :None
                [2023-12-08 00:37:28,507] [    INFO] - bf16                          :False
                [2023-12-08 00:37:28,507] [    INFO] - bf16_full_eval                :False
                [2023-12-08 00:37:28,507] [    INFO] - bias_correction               :False
                [2023-12-08 00:37:28,507] [    INFO] - current_device                :gpu:0
                [2023-12-08 00:37:28,507] [    INFO] - dataloader_drop_last          :False
                [2023-12-08 00:37:28,507] [    INFO] - dataloader_num_workers        :0
                [2023-12-08 00:37:28,507] [    INFO] - device                        :gpu
                [2023-12-08 00:37:28,507] [    INFO] - disable_tqdm                  :False
                [2023-12-08 00:37:28,507] [    INFO] - do_compress                   :False
                [2023-12-08 00:37:28,507] [    INFO] - do_eval                       :True
                [2023-12-08 00:37:28,508] [    INFO] - do_export                     :False
                [2023-12-08 00:37:28,508] [    INFO] - do_predict                    :False
                [2023-12-08 00:37:28,508] [    INFO] - do_train                      :False
                [2023-12-08 00:37:28,508] [    INFO] - eval_batch_size               :32
                [2023-12-08 00:37:28,508] [    INFO] - eval_steps                    :100
                [2023-12-08 00:37:28,508] [    INFO] - evaluation_strategy           :IntervalStrategy.STEPS
                [2023-12-08 00:37:28,508] [    INFO] - flatten_param_grads           :False
                [2023-12-08 00:37:28,508] [    INFO] - fp16                          :False
                [2023-12-08 00:37:28,508] [    INFO] - fp16_full_eval                :False
                [2023-12-08 00:37:28,508] [    INFO] - fp16_opt_level                :O1
                [2023-12-08 00:37:28,508] [    INFO] - gradient_accumulation_steps   :1
                [2023-12-08 00:37:28,508] [    INFO] - greater_is_better             :None
                [2023-12-08 00:37:28,508] [    INFO] - ignore_data_skip              :False
                [2023-12-08 00:37:28,508] [    INFO] - input_dtype                   :int64
                [2023-12-08 00:37:28,508] [    INFO] - input_infer_model_path        :None
                [2023-12-08 00:37:28,508] [    INFO] - label_names                   :None
                [2023-12-08 00:37:28,508] [    INFO] - lazy_data_processing          :True
                [2023-12-08 00:37:28,508] [    INFO] - learning_rate                 :5e-05
                [2023-12-08 00:37:28,508] [    INFO] - load_best_model_at_end        :False
                [2023-12-08 00:37:28,508] [    INFO] - local_process_index           :0
                [2023-12-08 00:37:28,508] [    INFO] - local_rank                    :-1
                [2023-12-08 00:37:28,508] [    INFO] - log_level                     :-1
                [2023-12-08 00:37:28,508] [    INFO] - log_level_replica             :-1
                [2023-12-08 00:37:28,508] [    INFO] - log_on_each_node              :True
                [2023-12-08 00:37:28,508] [    INFO] - logging_dir                   :checkpoint/runs/Dec08_00-37-25_jupyter-3980557-7016922
                [2023-12-08 00:37:28,509] [    INFO] - logging_first_step            :False
                [2023-12-08 00:37:28,509] [    INFO] - logging_steps                 :100
                [2023-12-08 00:37:28,509] [    INFO] - logging_strategy              :IntervalStrategy.STEPS
                [2023-12-08 00:37:28,509] [    INFO] - lr_scheduler_type             :SchedulerType.LINEAR
                [2023-12-08 00:37:28,509] [    INFO] - max_grad_norm                 :1.0
                [2023-12-08 00:37:28,509] [    INFO] - max_steps                     :-1
                [2023-12-08 00:37:28,509] [    INFO] - metric_for_best_model         :None
                [2023-12-08 00:37:28,509] [    INFO] - minimum_eval_times            :None
                [2023-12-08 00:37:28,509] [    INFO] - moving_rate                   :0.9
                [2023-12-08 00:37:28,509] [    INFO] - no_cuda                       :False
                [2023-12-08 00:37:28,509] [    INFO] - num_train_epochs              :3.0
                [2023-12-08 00:37:28,509] [    INFO] - onnx_format                   :True
                [2023-12-08 00:37:28,509] [    INFO] - optim                         :OptimizerNames.ADAMW
                [2023-12-08 00:37:28,509] [    INFO] - output_dir                    :checkpoint
                [2023-12-08 00:37:28,509] [    INFO] - overwrite_output_dir          :False
                [2023-12-08 00:37:28,509] [    INFO] - past_index                    :-1
                [2023-12-08 00:37:28,509] [    INFO] - per_device_eval_batch_size    :32
                [2023-12-08 00:37:28,509] [    INFO] - per_device_train_batch_size   :8
                [2023-12-08 00:37:28,509] [    INFO] - prediction_loss_only          :False
                [2023-12-08 00:37:28,509] [    INFO] - process_index                 :0
                [2023-12-08 00:37:28,509] [    INFO] - prune_embeddings              :False
                [2023-12-08 00:37:28,509] [    INFO] - recompute                     :False
                [2023-12-08 00:37:28,509] [    INFO] - remove_unused_columns         :True
                [2023-12-08 00:37:28,509] [    INFO] - report_to                     :['visualdl']
                [2023-12-08 00:37:28,509] [    INFO] - resume_from_checkpoint        :None
                [2023-12-08 00:37:28,509] [    INFO] - round_type                    :round
                [2023-12-08 00:37:28,510] [    INFO] - run_name                      :checkpoint
                [2023-12-08 00:37:28,510] [    INFO] - save_on_each_node             :False
                [2023-12-08 00:37:28,510] [    INFO] - save_steps                    :100
                [2023-12-08 00:37:28,510] [    INFO] - save_strategy                 :IntervalStrategy.STEPS
                [2023-12-08 00:37:28,510] [    INFO] - save_total_limit              :None
                [2023-12-08 00:37:28,510] [    INFO] - scale_loss                    :32768
                [2023-12-08 00:37:28,510] [    INFO] - seed                          :42
                [2023-12-08 00:37:28,510] [    INFO] - sharding                      :[]
                [2023-12-08 00:37:28,510] [    INFO] - sharding_degree               :-1
                [2023-12-08 00:37:28,510] [    INFO] - should_log                    :True
                [2023-12-08 00:37:28,510] [    INFO] - should_save                   :True
                [2023-12-08 00:37:28,510] [    INFO] - skip_memory_metrics           :True
                [2023-12-08 00:37:28,510] [    INFO] - strategy                      :dynabert+ptq
                [2023-12-08 00:37:28,510] [    INFO] - train_batch_size              :8
                [2023-12-08 00:37:28,510] [    INFO] - use_pact                      :True
                [2023-12-08 00:37:28,510] [    INFO] - warmup_ratio                  :0.1
                [2023-12-08 00:37:28,510] [    INFO] - warmup_steps                  :0
                [2023-12-08 00:37:28,510] [    INFO] - weight_decay                  :0.0
                [2023-12-08 00:37:28,510] [    INFO] - weight_quantize_type          :channel_wise_abs_max
                [2023-12-08 00:37:28,510] [    INFO] - width_mult_list               :None
                [2023-12-08 00:37:28,510] [    INFO] - world_size                    :1
                [2023-12-08 00:37:28,510] [    INFO] - 
                [2023-12-08 00:37:28,511] [    INFO] - ***** Running Prediction *****
                [2023-12-08 00:37:28,511] [    INFO] -   Num examples = 1000
                [2023-12-08 00:37:28,511] [    INFO] -   Total prediction steps = 32
                [2023-12-08 00:37:28,511] [    INFO] -   Pre device batch size = 32
                [2023-12-08 00:37:28,511] [    INFO] -   Total Batch size = 32
                 94%|████████████████████████████████████████▎  | 30/32 [00:03<00:00, 10.44it/s][2023-12-08 00:37:33,787] [    INFO] - -----Evaluate model-------
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
                [2023-12-08 00:37:33,788] [    INFO] - Evaluation examples in dev dataset: 55(5.5%) | precision: 47.62 | recall: 36.36 | F1 score 41.24
                [2023-12-08 00:37:33,788] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,788] [    INFO] - Class name: Other Amusement and Recreation Industries
                [2023-12-08 00:37:33,788] [    INFO] - Evaluation examples in dev dataset: 23(2.3%) | precision: 66.67 | recall: 69.57 | F1 score 68.09
                [2023-12-08 00:37:33,788] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,788] [    INFO] - Class name: Architectural  Engineering  and Related Services
                [2023-12-08 00:37:33,788] [    INFO] - Evaluation examples in dev dataset: 17(1.7%) | precision: 35.71 | recall: 58.82 | F1 score 44.44
                [2023-12-08 00:37:33,788] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,788] [    INFO] - Class name: Business  Professional  Labor  Political  and Similar Organizations
                [2023-12-08 00:37:33,788] [    INFO] - Evaluation examples in dev dataset: 19(1.9%) | precision: 33.33 | recall: 42.11 | F1 score 37.21
                [2023-12-08 00:37:33,788] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,788] [    INFO] - Class name: Offices of Other Health Practitioners
                [2023-12-08 00:37:33,788] [    INFO] - Evaluation examples in dev dataset: 22(2.2%) | precision: 52.00 | recall: 59.09 | F1 score 55.32
                [2023-12-08 00:37:33,788] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,788] [    INFO] - Class name: Motion Picture and Video Industries
                [2023-12-08 00:37:33,788] [    INFO] - Evaluation examples in dev dataset: 12(1.2%) | precision: 62.50 | recall: 83.33 | F1 score 71.43
                [2023-12-08 00:37:33,788] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,788] [    INFO] - Class name: Residential Building Construction
                [2023-12-08 00:37:33,788] [    INFO] - Evaluation examples in dev dataset: 25(2.5%) | precision: 76.00 | recall: 76.00 | F1 score 76.00
                [2023-12-08 00:37:33,788] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,788] [    INFO] - Class name: Civic and Social Organizations
                [2023-12-08 00:37:33,788] [    INFO] - Evaluation examples in dev dataset: 18(1.8%) | precision: 21.88 | recall: 38.89 | F1 score 28.00
                [2023-12-08 00:37:33,788] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,789] [    INFO] - Class name: Services to Buildings and Dwellings
                [2023-12-08 00:37:33,789] [    INFO] - Evaluation examples in dev dataset: 26(2.6%) | precision: 75.86 | recall: 84.62 | F1 score 80.00
                [2023-12-08 00:37:33,789] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,789] [    INFO] - Class name: Scientific Research and Development Services
                [2023-12-08 00:37:33,789] [    INFO] - Evaluation examples in dev dataset: 5(0.5%) | precision: 22.22 | recall: 40.00 | F1 score 28.57
                [2023-12-08 00:37:33,789] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,789] [    INFO] - Class name: Legal Services
                [2023-12-08 00:37:33,789] [    INFO] - Evaluation examples in dev dataset: 37(3.7%) | precision: 83.33 | recall: 81.08 | F1 score 82.19
                [2023-12-08 00:37:33,789] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,789] [    INFO] - Class name: Other Schools and Instruction
                [2023-12-08 00:37:33,789] [    INFO] - Evaluation examples in dev dataset: 10(1.0%) | precision: 38.10 | recall: 80.00 | F1 score 51.61
                [2023-12-08 00:37:33,789] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,789] [    INFO] - Class name: Personal and Household Goods Repair and Maintenance
                [2023-12-08 00:37:33,789] [    INFO] - Evaluation examples in dev dataset: 7(0.7%) | precision: 25.00 | recall: 28.57 | F1 score 26.67
                [2023-12-08 00:37:33,789] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,789] [    INFO] - Class name: Traveler Accommodation
                [2023-12-08 00:37:33,789] [    INFO] - Evaluation examples in dev dataset: 6(0.6%) | precision: 50.00 | recall: 16.67 | F1 score 25.00
                [2023-12-08 00:37:33,789] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,789] [    INFO] - Class name: Restaurants and Other Eating Places
                [2023-12-08 00:37:33,789] [    INFO] - Evaluation examples in dev dataset: 41(4.1%) | precision: 83.72 | recall: 87.80 | F1 score 85.71
                [2023-12-08 00:37:33,789] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,789] [    INFO] - Class name: Agencies  Brokerages  and Other Insurance Related Activities
                [2023-12-08 00:37:33,789] [    INFO] - Evaluation examples in dev dataset: 12(1.2%) | precision: 62.50 | recall: 83.33 | F1 score 71.43
                [2023-12-08 00:37:33,789] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,789] [    INFO] - Class name: Elementary and Secondary Schools
                [2023-12-08 00:37:33,789] [    INFO] - Evaluation examples in dev dataset: 28(2.8%) | precision: 86.96 | recall: 71.43 | F1 score 78.43
                [2023-12-08 00:37:33,789] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,789] [    INFO] - Class name: Outpatient Care Centers
                [2023-12-08 00:37:33,790] [    INFO] - Evaluation examples in dev dataset: 6(0.6%) | precision: 25.00 | recall: 33.33 | F1 score 28.57
                [2023-12-08 00:37:33,790] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,790] [    INFO] - Class name: Offices of Real Estate Agents and Brokers
                [2023-12-08 00:37:33,790] [    INFO] - Evaluation examples in dev dataset: 26(2.6%) | precision: 80.00 | recall: 61.54 | F1 score 69.57
                [2023-12-08 00:37:33,790] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,790] [    INFO] - Class name: Management of Companies and Enterprises
                [2023-12-08 00:37:33,790] [    INFO] - Evaluation examples in dev dataset: 5(0.5%) | precision: 25.00 | recall: 20.00 | F1 score 22.22
                [2023-12-08 00:37:33,790] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,790] [    INFO] - Class name: Other Specialty Trade Contractors
                [2023-12-08 00:37:33,790] [    INFO] - Evaluation examples in dev dataset: 4(0.4%) | precision: 9.09 | recall: 25.00 | F1 score 13.33
                [2023-12-08 00:37:33,790] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,790] [    INFO] - Class name: Individual and Family Services
                [2023-12-08 00:37:33,790] [    INFO] - Evaluation examples in dev dataset: 18(1.8%) | precision: 53.85 | recall: 38.89 | F1 score 45.16
                [2023-12-08 00:37:33,790] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,790] [    INFO] - Class name: Office Administrative Services
                [2023-12-08 00:37:33,790] [    INFO] - Evaluation examples in dev dataset: 9(0.9%) | precision: 40.00 | recall: 44.44 | F1 score 42.11
                [2023-12-08 00:37:33,790] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,790] [    INFO] - Class name: Employment Services
                [2023-12-08 00:37:33,790] [    INFO] - Evaluation examples in dev dataset: 3(0.3%) | precision: 33.33 | recall: 33.33 | F1 score 33.33
                [2023-12-08 00:37:33,790] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,790] [    INFO] - Class name: Social Advocacy Organizations
                [2023-12-08 00:37:33,790] [    INFO] - Evaluation examples in dev dataset: 9(0.9%) | precision: 12.50 | recall: 22.22 | F1 score 16.00
                [2023-12-08 00:37:33,790] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,790] [    INFO] - Class name: Offices of Dentists
                [2023-12-08 00:37:33,790] [    INFO] - Evaluation examples in dev dataset: 28(2.8%) | precision: 96.15 | recall: 89.29 | F1 score 92.59
                [2023-12-08 00:37:33,790] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,790] [    INFO] - Class name: Lessors of Real Estate
                [2023-12-08 00:37:33,790] [    INFO] - Evaluation examples in dev dataset: 6(0.6%) | precision: 19.05 | recall: 66.67 | F1 score 29.63
                [2023-12-08 00:37:33,791] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,791] [    INFO] - Class name: Religious Organizations
                [2023-12-08 00:37:33,791] [    INFO] - Evaluation examples in dev dataset: 27(2.7%) | precision: 90.48 | recall: 70.37 | F1 score 79.17
                [2023-12-08 00:37:33,791] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,791] [    INFO] - Class name: Land Subdivision
                [2023-12-08 00:37:33,791] [    INFO] - Evaluation examples in dev dataset: 3(0.3%) | precision: 33.33 | recall: 66.67 | F1 score 44.44
                [2023-12-08 00:37:33,791] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,791] [    INFO] - Class name: Offices of Physicians
                [2023-12-08 00:37:33,791] [    INFO] - Evaluation examples in dev dataset: 24(2.4%) | precision: 43.75 | recall: 58.33 | F1 score 50.00
                [2023-12-08 00:37:33,791] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,791] [    INFO] - Class name: Independent Artists  Writers  and Performers
                [2023-12-08 00:37:33,791] [    INFO] - Evaluation examples in dev dataset: 5(0.5%) | precision: 50.00 | recall: 40.00 | F1 score 44.44
                [2023-12-08 00:37:33,791] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,791] [    INFO] - Class name: Administration of Housing Programs  Urban Planning  and Community Development
                [2023-12-08 00:37:33,791] [    INFO] - Evaluation examples in dev dataset: 1(0.1%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,791] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,791] [    INFO] - Class name: Other Financial Investment Activities
                [2023-12-08 00:37:33,791] [    INFO] - Evaluation examples in dev dataset: 7(0.7%) | precision: 20.00 | recall: 28.57 | F1 score 23.53
                [2023-12-08 00:37:33,791] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,791] [    INFO] - Class name: Electronic and Precision Equipment Repair and Maintenance
                [2023-12-08 00:37:33,791] [    INFO] - Evaluation examples in dev dataset: 2(0.2%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,791] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,791] [    INFO] - Class name: Activities Related to Real Estate
                [2023-12-08 00:37:33,791] [    INFO] - Evaluation examples in dev dataset: 25(2.5%) | precision: 52.17 | recall: 48.00 | F1 score 50.00
                [2023-12-08 00:37:33,791] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,791] [    INFO] - Class name: Other Personal Services
                [2023-12-08 00:37:33,792] [    INFO] - Evaluation examples in dev dataset: 15(1.5%) | precision: 50.00 | recall: 60.00 | F1 score 54.55
                [2023-12-08 00:37:33,792] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,792] [    INFO] - Class name: Other Professional  Scientific  and Technical Services
                [2023-12-08 00:37:33,792] [    INFO] - Evaluation examples in dev dataset: 36(3.6%) | precision: 78.38 | recall: 80.56 | F1 score 79.45
                [2023-12-08 00:37:33,792] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,792] [    INFO] - Class name: Justice  Public Order  and Safety Activities
                [2023-12-08 00:37:33,792] [    INFO] - Evaluation examples in dev dataset: 2(0.2%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,792] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,792] [    INFO] - Class name: Computer Systems Design and Related Services
                [2023-12-08 00:37:33,792] [    INFO] - Evaluation examples in dev dataset: 22(2.2%) | precision: 45.16 | recall: 63.64 | F1 score 52.83
                [2023-12-08 00:37:33,792] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,792] [    INFO] - Class name: Building Equipment Contractors
                [2023-12-08 00:37:33,792] [    INFO] - Evaluation examples in dev dataset: 10(1.0%) | precision: 43.75 | recall: 70.00 | F1 score 53.85
                [2023-12-08 00:37:33,792] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,792] [    INFO] - Class name: Other Heavy and Civil Engineering Construction
                [2023-12-08 00:37:33,792] [    INFO] - Evaluation examples in dev dataset: 0 (0%)
                [2023-12-08 00:37:33,792] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,792] [    INFO] - Class name: Miscellaneous Durable Goods Merchant Wholesalers
                [2023-12-08 00:37:33,792] [    INFO] - Evaluation examples in dev dataset: 5(0.5%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,792] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,792] [    INFO] - Class name: Community Food and Housing  and Emergency and Other Relief Services
                [2023-12-08 00:37:33,792] [    INFO] - Evaluation examples in dev dataset: 1(0.1%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,792] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,792] [    INFO] - Class name: Hardware  and Plumbing and Heating Equipment and Supplies Merchant Wholesalers
                [2023-12-08 00:37:33,792] [    INFO] - Evaluation examples in dev dataset: 1(0.1%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,792] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,792] [    INFO] - Class name: Professional and Commercial Equipment and Supplies Merchant Wholesalers
                [2023-12-08 00:37:33,792] [    INFO] - Evaluation examples in dev dataset: 5(0.5%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,793] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,793] [    INFO] - Class name: Commercial and Industrial Machinery and Equipment (except Automotive and Electronic) Repair and Maintenance
                [2023-12-08 00:37:33,793] [    INFO] - Evaluation examples in dev dataset: 2(0.2%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,793] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,793] [    INFO] - Class name: Farm Product Raw Material Merchant Wholesalers
                [2023-12-08 00:37:33,793] [    INFO] - Evaluation examples in dev dataset: 2(0.2%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,793] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,793] [    INFO] - Class name: Automotive Repair and Maintenance
                [2023-12-08 00:37:33,793] [    INFO] - Evaluation examples in dev dataset: 19(1.9%) | precision: 69.57 | recall: 84.21 | F1 score 76.19
                [2023-12-08 00:37:33,793] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,793] [    INFO] - Class name: Machinery  Equipment  and Supplies Merchant Wholesalers
                [2023-12-08 00:37:33,793] [    INFO] - Evaluation examples in dev dataset: 3(0.3%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,793] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,793] [    INFO] - Class name: Drinking Places (Alcoholic Beverages)
                [2023-12-08 00:37:33,793] [    INFO] - Evaluation examples in dev dataset: 7(0.7%) | precision: 62.50 | recall: 71.43 | F1 score 66.67
                [2023-12-08 00:37:33,793] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,793] [    INFO] - Class name: Grantmaking and Giving Services
                [2023-12-08 00:37:33,793] [    INFO] - Evaluation examples in dev dataset: 4(0.4%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,793] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,793] [    INFO] - Class name: Nondepository Credit Intermediation
                [2023-12-08 00:37:33,793] [    INFO] - Evaluation examples in dev dataset: 6(0.6%) | precision: 55.56 | recall: 83.33 | F1 score 66.67
                [2023-12-08 00:37:33,793] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,793] [    INFO] - Class name: Child Day Care Services
                [2023-12-08 00:37:33,793] [    INFO] - Evaluation examples in dev dataset: 6(0.6%) | precision: 80.00 | recall: 66.67 | F1 score 72.73
                [2023-12-08 00:37:33,793] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,793] [    INFO] - Class name: Commercial and Industrial Machinery and Equipment Rental and Leasing
                [2023-12-08 00:37:33,793] [    INFO] - Evaluation examples in dev dataset: 4(0.4%) | precision: 33.33 | recall: 25.00 | F1 score 28.57
                [2023-12-08 00:37:33,793] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,794] [    INFO] - Class name: Household Appliances and Electrical and Electronic Goods Merchant Wholesalers
                [2023-12-08 00:37:33,794] [    INFO] - Evaluation examples in dev dataset: 3(0.3%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,794] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,794] [    INFO] - Class name: Travel Arrangement and Reservation Services
                [2023-12-08 00:37:33,794] [    INFO] - Evaluation examples in dev dataset: 5(0.5%) | precision: 100.00 | recall: 80.00 | F1 score 88.89
                [2023-12-08 00:37:33,794] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,794] [    INFO] - Class name: Nursing Care Facilities (Skilled Nursing Facilities)
                [2023-12-08 00:37:33,794] [    INFO] - Evaluation examples in dev dataset: 4(0.4%) | precision: 25.00 | recall: 25.00 | F1 score 25.00
                [2023-12-08 00:37:33,794] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,794] [    INFO] - Class name: Petroleum and Petroleum Products Merchant Wholesalers
                [2023-12-08 00:37:33,794] [    INFO] - Evaluation examples in dev dataset: 3(0.3%) | precision: 33.33 | recall: 33.33 | F1 score 33.33
                [2023-12-08 00:37:33,794] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,794] [    INFO] - Class name: Promoters of Performing Arts  Sports  and Similar Events
                [2023-12-08 00:37:33,794] [    INFO] - Evaluation examples in dev dataset: 4(0.4%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,794] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,794] [    INFO] - Class name: General Medical and Surgical Hospitals
                [2023-12-08 00:37:33,794] [    INFO] - Evaluation examples in dev dataset: 5(0.5%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,794] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,794] [    INFO] - Class name: Accounting  Tax Preparation  Bookkeeping  and Payroll Services
                [2023-12-08 00:37:33,794] [    INFO] - Evaluation examples in dev dataset: 7(0.7%) | precision: 66.67 | recall: 57.14 | F1 score 61.54
                [2023-12-08 00:37:33,794] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,794] [    INFO] - Class name: Foundation  Structure  and Building Exterior Contractors
                [2023-12-08 00:37:33,794] [    INFO] - Evaluation examples in dev dataset: 14(1.4%) | precision: 64.29 | recall: 64.29 | F1 score 64.29
                [2023-12-08 00:37:33,794] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,794] [    INFO] - Class name: Support Activities for Mining
                [2023-12-08 00:37:33,794] [    INFO] - Evaluation examples in dev dataset: 1(0.1%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,794] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,794] [    INFO] - Class name: Nonresidential Building Construction
                [2023-12-08 00:37:33,795] [    INFO] - Evaluation examples in dev dataset: 5(0.5%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,795] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,795] [    INFO] - Class name: Specialized Design Services
                [2023-12-08 00:37:33,795] [    INFO] - Evaluation examples in dev dataset: 20(2.0%) | precision: 42.31 | recall: 55.00 | F1 score 47.83
                [2023-12-08 00:37:33,795] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,795] [    INFO] - Class name: Greenhouse  Nursery  and Floriculture Production
                [2023-12-08 00:37:33,795] [    INFO] - Evaluation examples in dev dataset: 2(0.2%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,795] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,795] [    INFO] - Class name: RV (Recreational Vehicle) Parks and Recreational Camps
                [2023-12-08 00:37:33,795] [    INFO] - Evaluation examples in dev dataset: 4(0.4%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,795] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,795] [    INFO] - Class name: Oilseed and Grain Farming
                [2023-12-08 00:37:33,795] [    INFO] - Evaluation examples in dev dataset: 1(0.1%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,795] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,795] [    INFO] - Class name: Miscellaneous Nondurable Goods Merchant Wholesalers
                [2023-12-08 00:37:33,795] [    INFO] - Evaluation examples in dev dataset: 10(1.0%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,795] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,795] [    INFO] - Class name: Lumber and Other Construction Materials Merchant Wholesalers
                [2023-12-08 00:37:33,795] [    INFO] - Evaluation examples in dev dataset: 1(0.1%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,795] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,795] [    INFO] - Class name: Water  Sewage and Other Systems
                [2023-12-08 00:37:33,795] [    INFO] - Evaluation examples in dev dataset: 2(0.2%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,795] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,795] [    INFO] - Class name: Other Investment Pools and Funds
                [2023-12-08 00:37:33,795] [    INFO] - Evaluation examples in dev dataset: 1(0.1%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,795] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,795] [    INFO] - Class name: Vocational Rehabilitation Services
                [2023-12-08 00:37:33,796] [    INFO] - Evaluation examples in dev dataset: 1(0.1%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,796] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,796] [    INFO] - Class name: Grocery and Related Product Merchant Wholesalers
                [2023-12-08 00:37:33,796] [    INFO] - Evaluation examples in dev dataset: 5(0.5%) | precision: 50.00 | recall: 20.00 | F1 score 28.57
                [2023-12-08 00:37:33,796] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,796] [    INFO] - Class name: Waste Collection
                [2023-12-08 00:37:33,796] [    INFO] - Evaluation examples in dev dataset: 1(0.1%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,796] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,796] [    INFO] - Class name: Museums  Historical Sites  and Similar Institutions
                [2023-12-08 00:37:33,796] [    INFO] - Evaluation examples in dev dataset: 6(0.6%) | precision: 50.00 | recall: 50.00 | F1 score 50.00
                [2023-12-08 00:37:33,796] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,796] [    INFO] - Class name: Consumer Goods Rental
                [2023-12-08 00:37:33,796] [    INFO] - Evaluation examples in dev dataset: 3(0.3%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,796] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,796] [    INFO] - Class name: Activities Related to Credit Intermediation
                [2023-12-08 00:37:33,796] [    INFO] - Evaluation examples in dev dataset: 4(0.4%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,796] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,796] [    INFO] - Class name: Junior Colleges
                [2023-12-08 00:37:33,796] [    INFO] - Evaluation examples in dev dataset: 2(0.2%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,796] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,796] [    INFO] - Class name: Performing Arts Companies
                [2023-12-08 00:37:33,796] [    INFO] - Evaluation examples in dev dataset: 2(0.2%) | precision: 50.00 | recall: 50.00 | F1 score 50.00
                [2023-12-08 00:37:33,796] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,796] [    INFO] - Class name: Other Animal Production
                [2023-12-08 00:37:33,796] [    INFO] - Evaluation examples in dev dataset: 5(0.5%) | precision: 66.67 | recall: 40.00 | F1 score 50.00
                [2023-12-08 00:37:33,797] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,797] [    INFO] - Class name: Executive  Legislative  and Other General Government Support
                [2023-12-08 00:37:33,797] [    INFO] - Evaluation examples in dev dataset: 6(0.6%) | precision: 57.14 | recall: 66.67 | F1 score 61.54
                [2023-12-08 00:37:33,797] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,797] [    INFO] - Class name: Other Support Services
                [2023-12-08 00:37:33,797] [    INFO] - Evaluation examples in dev dataset: 13(1.3%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,797] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,797] [    INFO] - Class name: Spectator Sports
                [2023-12-08 00:37:33,797] [    INFO] - Evaluation examples in dev dataset: 1(0.1%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,797] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,797] [    INFO] - Class name: Residential Intellectual and Developmental Disability  Mental Health  and Substance Abuse Facilities
                [2023-12-08 00:37:33,797] [    INFO] - Evaluation examples in dev dataset: 1(0.1%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,797] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,797] [    INFO] - Class name: Building Finishing Contractors
                [2023-12-08 00:37:33,797] [    INFO] - Evaluation examples in dev dataset: 10(1.0%) | precision: 46.15 | recall: 60.00 | F1 score 52.17
                [2023-12-08 00:37:33,797] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,797] [    INFO] - Class name: Investigation and Security Services
                [2023-12-08 00:37:33,797] [    INFO] - Evaluation examples in dev dataset: 8(0.8%) | precision: 85.71 | recall: 75.00 | F1 score 80.00
                [2023-12-08 00:37:33,797] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,797] [    INFO] - Class name: Technical and Trade Schools
                [2023-12-08 00:37:33,797] [    INFO] - Evaluation examples in dev dataset: 3(0.3%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,797] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,797] [    INFO] - Class name: Home Health Care Services
                [2023-12-08 00:37:33,797] [    INFO] - Evaluation examples in dev dataset: 3(0.3%) | precision: 25.00 | recall: 33.33 | F1 score 28.57
                [2023-12-08 00:37:33,797] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,797] [    INFO] - Class name: Business Support Services
                [2023-12-08 00:37:33,797] [    INFO] - Evaluation examples in dev dataset: 2(0.2%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,797] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,798] [    INFO] - Class name: Other Crop Farming
                [2023-12-08 00:37:33,798] [    INFO] - Evaluation examples in dev dataset: 7(0.7%) | precision: 33.33 | recall: 57.14 | F1 score 42.11
                [2023-12-08 00:37:33,798] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,798] [    INFO] - Class name: Waste Treatment and Disposal
                [2023-12-08 00:37:33,798] [    INFO] - Evaluation examples in dev dataset: 1(0.1%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,798] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,798] [    INFO] - Class name: Highway  Street  and Bridge Construction
                [2023-12-08 00:37:33,798] [    INFO] - Evaluation examples in dev dataset: 2(0.2%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,798] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,798] [    INFO] - Class name: Depository Credit Intermediation
                [2023-12-08 00:37:33,798] [    INFO] - Evaluation examples in dev dataset: 6(0.6%) | precision: 100.00 | recall: 50.00 | F1 score 66.67
                [2023-12-08 00:37:33,798] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,798] [    INFO] - Class name: Advertising  Public Relations  and Related Services
                [2023-12-08 00:37:33,798] [    INFO] - Evaluation examples in dev dataset: 5(0.5%) | precision: 25.00 | recall: 40.00 | F1 score 30.77
                [2023-12-08 00:37:33,798] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,798] [    INFO] - Class name: Paper and Paper Product Merchant Wholesalers
                [2023-12-08 00:37:33,798] [    INFO] - Evaluation examples in dev dataset: 2(0.2%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,798] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,798] [    INFO] - Class name: Motor Vehicle and Motor Vehicle Parts and Supplies Merchant Wholesalers
                [2023-12-08 00:37:33,798] [    INFO] - Evaluation examples in dev dataset: 2(0.2%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,798] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,798] [    INFO] - Class name: Other Residential Care Facilities
                [2023-12-08 00:37:33,798] [    INFO] - Evaluation examples in dev dataset: 2(0.2%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,798] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,798] [    INFO] - Class name: Furniture and Home Furnishing Merchant Wholesalers
                [2023-12-08 00:37:33,798] [    INFO] - Evaluation examples in dev dataset: 4(0.4%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,798] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,798] [    INFO] - Class name: Educational Support Services
                [2023-12-08 00:37:33,799] [    INFO] - Evaluation examples in dev dataset: 4(0.4%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,799] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,799] [    INFO] - Class name: Vegetable and Melon Farming
                [2023-12-08 00:37:33,799] [    INFO] - Evaluation examples in dev dataset: 1(0.1%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,799] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,799] [    INFO] - Class name: Utility System Construction
                [2023-12-08 00:37:33,799] [    INFO] - Evaluation examples in dev dataset: 1(0.1%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,799] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,799] [    INFO] - Class name: Metal and Mineral (except Petroleum) Merchant Wholesalers
                [2023-12-08 00:37:33,799] [    INFO] - Evaluation examples in dev dataset: 1(0.1%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,799] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,799] [    INFO] - Class name: Death Care Services
                [2023-12-08 00:37:33,799] [    INFO] - Evaluation examples in dev dataset: 3(0.3%) | precision: 100.00 | recall: 100.00 | F1 score 100.00
                [2023-12-08 00:37:33,799] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,799] [    INFO] - Class name: Apparel  Piece Goods  and Notions Merchant Wholesalers
                [2023-12-08 00:37:33,799] [    INFO] - Evaluation examples in dev dataset: 2(0.2%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,799] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,799] [    INFO] - Class name: Administration of Environmental Quality Programs
                [2023-12-08 00:37:33,799] [    INFO] - Evaluation examples in dev dataset: 1(0.1%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,799] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,799] [    INFO] - Class name: Electric Power Generation  Transmission and Distribution
                [2023-12-08 00:37:33,799] [    INFO] - Evaluation examples in dev dataset: 3(0.3%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,799] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,799] [    INFO] - Class name: Support Activities for Animal Production
                [2023-12-08 00:37:33,799] [    INFO] - Evaluation examples in dev dataset: 1(0.1%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,799] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,799] [    INFO] - Class name: Colleges  Universities  and Professional Schools
                [2023-12-08 00:37:33,800] [    INFO] - Evaluation examples in dev dataset: 4(0.4%) | precision: 33.33 | recall: 50.00 | F1 score 40.00
                [2023-12-08 00:37:33,800] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,800] [    INFO] - Class name: Agents and Managers for Artists  Athletes  Entertainers  and Other Public Figures
                [2023-12-08 00:37:33,800] [    INFO] - Evaluation examples in dev dataset: 1(0.1%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,800] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,800] [    INFO] - Class name: Insurance Carriers
                [2023-12-08 00:37:33,800] [    INFO] - Evaluation examples in dev dataset: 1(0.1%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,800] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,800] [    INFO] - Class name: Business Schools and Computer and Management Training
                [2023-12-08 00:37:33,800] [    INFO] - Evaluation examples in dev dataset: 1(0.1%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,800] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,800] [    INFO] - Class name: Automotive Equipment Rental and Leasing
                [2023-12-08 00:37:33,800] [    INFO] - Evaluation examples in dev dataset: 1(0.1%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,800] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,800] [    INFO] - Class name: Medical and Diagnostic Laboratories
                [2023-12-08 00:37:33,800] [    INFO] - Evaluation examples in dev dataset: 1(0.1%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,800] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,800] [    INFO] - Class name: Remediation and Other Waste Management Services
                [2023-12-08 00:37:33,800] [    INFO] - Evaluation examples in dev dataset: 1(0.1%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,800] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,800] [    INFO] - Class name: Special Food Services
                [2023-12-08 00:37:33,800] [    INFO] - Evaluation examples in dev dataset: 0(0.0%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,800] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,800] [    INFO] - Class name: Continuing Care Retirement Communities and Assisted Living Facilities for the Elderly
                [2023-12-08 00:37:33,800] [    INFO] - Evaluation examples in dev dataset: 0 (0%)
                [2023-12-08 00:37:33,800] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,800] [    INFO] - Class name: Data Processing  Hosting  and Related Services
                [2023-12-08 00:37:33,800] [    INFO] - Evaluation examples in dev dataset: 0 (0%)
                [2023-12-08 00:37:33,801] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,801] [    INFO] - Class name: Administration of Economic Programs
                [2023-12-08 00:37:33,801] [    INFO] - Evaluation examples in dev dataset: 0 (0%)
                [2023-12-08 00:37:33,801] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,801] [    INFO] - Class name: Drycleaning and Laundry Services
                [2023-12-08 00:37:33,801] [    INFO] - Evaluation examples in dev dataset: 0(0.0%) | precision: 0.00 | recall: 0.00 | F1 score 0.00
                [2023-12-08 00:37:33,801] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,801] [    INFO] - Class name: Support Activities for Crop Production
                [2023-12-08 00:37:33,801] [    INFO] - Evaluation examples in dev dataset: 0 (0%)
                [2023-12-08 00:37:33,801] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,801] [    INFO] - Class name: Drugs and Druggists' Sundries Merchant Wholesalers
                [2023-12-08 00:37:33,801] [    INFO] - Evaluation examples in dev dataset: 0 (0%)
                [2023-12-08 00:37:33,801] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,801] [    INFO] - Class name: Chemical and Allied Products Merchant Wholesalers
                [2023-12-08 00:37:33,801] [    INFO] - Evaluation examples in dev dataset: 0 (0%)
                [2023-12-08 00:37:33,801] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,801] [    INFO] - Class name: Fruit and Tree Nut Farming
                [2023-12-08 00:37:33,801] [    INFO] - Evaluation examples in dev dataset: 0 (0%)
                [2023-12-08 00:37:33,801] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,801] [    INFO] - Class name: Poultry and Egg Production
                [2023-12-08 00:37:33,801] [    INFO] - Evaluation examples in dev dataset: 0 (0%)
                [2023-12-08 00:37:33,801] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,801] [    INFO] - Class name: Psychiatric and Substance Abuse Hospitals
                [2023-12-08 00:37:33,801] [    INFO] - Evaluation examples in dev dataset: 0 (0%)
                [2023-12-08 00:37:33,801] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,801] [    INFO] - Class name: Specialty (except Psychiatric and Substance Abuse) Hospitals
                [2023-12-08 00:37:33,801] [    INFO] - Evaluation examples in dev dataset: 0 (0%)
                [2023-12-08 00:37:33,801] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,801] [    INFO] - Class name: Nonmetallic Mineral Mining and Quarrying
                [2023-12-08 00:37:33,802] [    INFO] - Evaluation examples in dev dataset: 0 (0%)
                [2023-12-08 00:37:33,802] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,802] [    INFO] - Class name: Sound Recording Industries
                [2023-12-08 00:37:33,802] [    INFO] - Evaluation examples in dev dataset: 0 (0%)
                [2023-12-08 00:37:33,802] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,802] [    INFO] - Class name: Administration of Human Resource Programs
                [2023-12-08 00:37:33,802] [    INFO] - Evaluation examples in dev dataset: 0 (0%)
                [2023-12-08 00:37:33,802] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,802] [    INFO] - Class name: Cattle Ranching and Farming
                [2023-12-08 00:37:33,802] [    INFO] - Evaluation examples in dev dataset: 0 (0%)
                [2023-12-08 00:37:33,802] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,802] [    INFO] - Class name: Wholesale Electronic Markets and Agents and Brokers
                [2023-12-08 00:37:33,802] [    INFO] - Evaluation examples in dev dataset: 0 (0%)
                [2023-12-08 00:37:33,802] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,802] [    INFO] - Class name: Natural Gas Distribution
                [2023-12-08 00:37:33,802] [    INFO] - Evaluation examples in dev dataset: 0 (0%)
                [2023-12-08 00:37:33,802] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,802] [    INFO] - Class name: Amusement Parks and Arcades
                [2023-12-08 00:37:33,802] [    INFO] - Evaluation examples in dev dataset: 0 (0%)
                [2023-12-08 00:37:33,802] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,802] [    INFO] - Class name: Securities and Commodity Exchanges
                [2023-12-08 00:37:33,802] [    INFO] - Evaluation examples in dev dataset: 0 (0%)
                [2023-12-08 00:37:33,802] [    INFO] - ----------------------------
                [2023-12-08 00:37:33,804] [    INFO] - Bad case in dev dataset saved in ./data/bad_case.txt
                100%|███████████████████████████████████████████| 32/32 [00:04<00:00,  7.77it/s]


<img src="https://github.com/Fent1/Multi_Classification/assets/43925272/b22b5893-5585-4d87-a52f-f127c06507dc" alt="image" width="300" height="auto">

 - 对于这个数据集，模型性能很好，与之前的模型相比，准确率从30.42%增加到55.60%。
 - 精确率和召回率都很低，分别为32.37和35.51，这意味着这个分类模型仍然丢失一些特征的获取。
 - 由于每个标签可能存在严重的不平衡，我们的模型性能高但F1分数低。

## 如何运行

 1. 请先运行网络爬虫代码，请直接在Business_Industry_URLS.csv同目录下运行Starter_Web_Scraper_2.py。
 2. 运行完成后会生Business_Industry_URLS_wText.csv文件，请在main.py中修改train_path对应的文件地址。
 3. 运行main.py。

P.S: 由于某些原因现在还没有上传数据集Business_Industry_URLS.csv
