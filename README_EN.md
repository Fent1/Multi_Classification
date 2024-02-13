# Multi-classification Task
[Chinese](./README_EN.md) | English
## Background
Pychex company wants to gain a more detailed understanding of their target customer base and has provided us with a list of website addresses for over 10,000 companies, along with their company types. Pychex hopes that we can build a model that accurately classifies the type of company based on the provided company website.

## Dataset
In the dataset of over 10,000 company website entries, there are a total of 17 types of companies, as shown below:

## Web Scraping
Since the provided dataset contains website links, we need to use web scraping to extract usable information from the websites and generate a new Text column as an independent variable for training the model. Due to the extensive nature of scraping over 10,000 websites, we adopted a multiprocessing approach to leverage multiple CPUs for computation and information retrieval, reducing the scraping time to 6 hours.

## Model Selection
For the selection of models for multi-classification problems, BERT and ERNIE are both mature frameworks that are widely used. There isn't a significant difference in accuracy between the two models after training, so we chose to train the model using the ERNIE framework.

## Model Performance

- The model performs well on this dataset, with accuracy increasing from 30.42% to 55.60% compared to the previous model.
- Precision and recall rates are both low, at 32.37 and 35.51 respectively, indicating that this classification model still misses some feature extraction.
- Due to the potential serious imbalance in each label, our model performs well but has a low F1 score.

## How to Run

1. Please run the web scraping code first by running Starter_Web_Scraper_2.py directly in the same directory as Business_Industry_URLS.csv.
2. After completion, a Business_Industry_URLS_wText.csv file will be generated. Modify the train_path in main.py to point to the corresponding file address.
3. Run main.py.
