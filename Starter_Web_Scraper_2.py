# Summary
"""
This file is intended to help you speed up the web scraping step in this project, but you will have to edit parameters
and even adjust pieces of the code to create and prepare a good dataset to model on.

This script uses multiprocessing to efficiently scrape text from the websites specified in the data file provided. A key
feature of this file is that it writes scraped data as it goes, so that if the process is cancelled or interrupted, the
scraping progress has not been lost. If the process is interupted for any reason, there will be some urls that were in
the process of being written, that will show up in the final file as partial data, and will need to be filtered out. 

We also provide a requirements.txt file and pip freeze to help get this script working on your computer.
"""

# Imports ##############################################################

import requests # executing HTTP requests
requests.packages.urllib3.disable_warnings()  # supress warnings if needed for verify=False
from bs4 import BeautifulSoup, SoupStrainer # parsing and searching web contents like html/xml
from bs4.element import Comment
import pandas as pd # data manipulation
import numpy as np
import csv # writing to csv file
import re # regular expressions to remove any unwanted characters in website text
from sklearn import feature_extraction # for stop words
import multiprocessing as mp # for faster processing taking advance of multiple cores
from time import time
import os
import sys

###############################################################

# Global features ##############################################################

file_in = 'Business_Industry_URLS.csv'
file_out = 'Business_Industry_URLS_wText.csv'

""" 
slw: sub link words
We may not want to scrape all of the pages on a website, but only pages we're interested in. This list 
is used to only scrape the pages that contain one of these words in the url. You may want to improve
this list.
"""
slw = ['about', 'services', 'company', 'business',  'clients',
            'information', 'missions', 'who-we-are', 'what-we-do', 'our-story',
            'faq', 'questions-about-us', 'overview', 'background',
            'goal', 'objectives', 'activities', 'our', 'summary', 'introduction',
            'updates', 'testimonials', 'Gallery', 'get-to-know-us', 'what-we-do',
            'What-I-Cover', 'our-services', "view-details","forums","Affiliate-News","benefits","office","membership","toolkit","general-resources","Affiliate-resources",
       "Featured-Topics","Featured-Facilities","learn-more","Type","special-focus","mental-health", "our-service","look-here","map","promotions","faqs",
       "RSSING","Press", "Releases","latest","popular","top-rated","trending","browsing-latest-articles",
       "Ask","Coach","Professional-sports-Podcast","Speaking-engagements","Books","about","listen","browse-books","best-of-the-blog",
       "services","about","portfolio","learn-more", "Virtual-events","Planning", "resources","our-Brands","what-size-unit","who-we-are",
       "NEWS","live-feed","events", "get-to-know-us","borrow-for-your-business","housing-and-homeownership","entrepreneurship-services","grow-your-community","our-success-stories","publications",
       "buy-a-domain","sell-your-domain", "our-mission","What-are-Our-Customers-Saying","rentals","operations",
       "equipment","area-of-operation","current-activities", "what-we-do","brochure","insights","news",
       "floor-plans","amenities","neighborhood", "airport-services","fleet","corporate-accounts",
       "commercial","residential", "Testimonial", "Explore", "company","training", "read-article", "experimental-research","medical-device-design","prototypes","government","embedded-systems","strategy-discovery","process",
       "grads","projects","purpose", "your-choice","programs","education","news&agendas","how-it-works","why-choose",
       "department", "minutes","highligts", "why-volunteer","Community","posts", "cabins","rates","park-rules",
       "features", "properties","Why-Choose-Us", "about-me","What-I-Cover", "announcements", "tips","read-all-reviews",
       "alternative-energy", "rates&quotes", "customers","search-products","our-services", "history","industries",
       "media","our-story", "techinical","catalog"]

log_name = 'web_scraper_log.txt' # log file path to write out

"""
Number of business urls you want to load in for text scraping. The file have 100,000.
It's up to you how many you need or want to build a model! You will definitely want more 
than 1000!
"""
num_urls = 10000

"""
This is the number of data batches to split the original data into, for parallel (multi) processing.
Whenever a batch finishes, the scraped text data will be saved and the progress tracker will be updated.
You may want to change this number depending on how many URLs you decide to use. Having <100 URLs for
each batch will make each batch finish faster, while having >100 URLs will make each batch take longer. 
It's recommended to use 50-150 URLs per batch to make sure progress is saved, but that the script doesn't waste
time managing the overhead of many batches. 
"""
num_batches = 150

sw = feature_extraction.text.ENGLISH_STOP_WORDS # sw: stop words (remove stop words in text processing later)

###############################################################

# Scraping functions ##############################################################

def get_text(url):
    # h: headers
    h = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) ' +
                          'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36'
    }
    """
    r: response ()
       
    If you get an error about SSL Certificate validation, you can consider setting verify=False, this disables SSL 
    Certificate verification. SSL Certificates allow web browsers to identify and establish encrypted network connections 
    to web sites using the SSL/TLS protocol. SSL/TLS is used to protect the data exchanged between the endpoints 
    (i.e. your scraper and the web servers) against sniffing and tampering. If no sensitive data are transmitted and you 
    don't care about somebody sniffing or modifying the data, then you can disable certificate validation. Since we are 
    making requests to public information in the html of business webpages, there should be no sensitive data being transmitted 
    that would be a concern for sniffing or tampering, making this an acceptable practice.
    """
    r = requests.get(url, allow_redirects=True, verify=False, headers=h) # request the URL
       
    # scrape html and xml differently, and don't scrape other files
    # ft: file type
    ft = r.headers['content-type'].split(';')[0].split('/')[1]
    if ft == 'html':
        # s: soup
        s = BeautifulSoup(r.text, "html.parser") # get contents of the webpage based on Beautifulsoup
        # t: text
        t = s.findAll('p', string=True) # use findAll selector to get text
        return r, s, t
    else:
        return r, None, ''
    
    

"""
Takes a url and returns scraped text for the domain and desired sublinks. It is likely that this
function will have to be improved to get the exact text you're interested in for modeling.
"""
def scrape_page(url):   
    # wrap in a try-except to catch any errors if they occur
    try:
        # ts: texts
        ts = []
        # rs: responses
        rs = []
        # slf: sub links formatted for later
        slf = []
        
        # dt: domain text
        r, s, dt = get_text(url)
        if s is None:
            return ts, slf, rs
        else:
            rs = rs + [r.status_code]
            ts.append(dt)

            # asl: all sub links
            asl = [l['href'] for l in s.findAll('a', href=True)]  # use findAll selector to get sublinks
            
            # sl: sub links (filter the links that we want)
            sl = [x for x in asl if any(word in x for word in slw)]
            
            # l :link
            for l in sl:
                if l.startswith('/'):
                    slf = slf + [url+l]
                elif l.startswith(url): 
                    slf = slf + [l]
                else:
                    pass

            slf = list(dict.fromkeys(slf))  # remove duplicated sublinks
            
            # get sublink texts
            for l in slf:
                r, _, t = get_text(l)
                rs = rs + [r.status_code]
                ts.append(t)
                
                   
            return ts, slf, rs
        
    except Exception as e:
        
        return ts, slf, rs

# Returns true if the html text elements are visible on the webpage for reading, otherwise false
def tag_is_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

"""
Takes the raw scraped text for a given website and applies many different transformations and filters 
to prepare the text for encoding and modeling. It is expected that this will have to be modified to
get the text data and text size that you're interested in.
"""
def process_text(ts):
    # formalize page contents
    #ts = filter(tag_is_visible, ts)  # parse and get visible t
    #ts = np.array([t.strip() for t in ts])  # strip off spaces from ends
    #ts = ts[ts.astype(str) != '']  # remove blanks
    #ts = [t.lower() for t in ts]
    #ts = list(dict.fromkeys(ts))  # remove duplication
    #ts = ' '.join(ts) # all into one string
    
    # Detect if web scraping is prohibited, if True, do not return any words
    if 'web scraping' in ts:
        ts = ''
    else:
        # preProcess strings
        ts = ts.split()  # split string into words
        # ts = [re.sub(r'[^a-zA-Z]', '', x) for x in ts]  # only keep English word
        ts = [x for x in ts if x != '']  # remove blanks
        # ts = [x for x in ts if x not in sw]  # remove stopwords
        ts = " ".join(ts) # join back into one string
        ts = ts.replace("\n", "")
        ts = ts.replace("\r", "")
        ts = ts.replace("\xa0", "")
        ts = ts.replace("\t", "")
        if len(ts.split(' ')) <= 10 or len(ts.split(' ')) >= 1000:
            ts = ''

    return ts

def format_url(url):
    if url[:8] != 'https://':
        url = 'https://' + url
    return url
    
###############################################################

# Multiprocessing functions ##############################################################

"""
The workers are the multiple processes that actually do the work on the batches. When they finish 
one batch, they take another from a queue until all the batches have been processed. If you add
any functions or steps outside of the preexisting functions, then this is where you would add
code to tell the workers what to do. You will need to intelligently edit line 204 below, to specify 
how long of a string you want to save, depending on how much text you want to scrape for each website.
"""
def worker(arg):
    batch = batches[arg]
    m = f'Process {arg} Start\n'
    # work below------------------------------------------------------------
    texts = []
    for row in batch.iterrows():
        url= row[1].iloc[2]
        # get text from domain and all chosen sub links
        ts, _, _ = scrape_page(url)
        # tsp: texts processed
        tsp = []
        # processes each page and concat into one long string
        count = 0
        for t in ts:
            # tp: text processed
            for p in t:
                tp = process_text(p.get_text())
                if tp != '':
                    tsp = tsp + [tp]
                    break
            break
        # tl: text long
        if len(tsp) == 0:
            tl = ''
        else:
            tl = ' '.join(tsp)[:1000] # can be very long, may want to reduce size in some intelligent way
        texts = texts + [tl]
    print(texts)
    batch_out = batch.assign(Text=texts)
    
    return batch_out
            
###############################################################

# Initialization ##############################################################

# load data
data = pd.read_csv(file_in, nrows=num_urls) # how many urls to load in
# open out file for async writing and write column headers
fo = open(file_out,'w', encoding='ascii', errors='ignore', newline='')
writer = csv.writer(fo, delimiter=',')
writer.writerow(list(data.columns) + ['Text']) 

# formalize urls
data['URL'] = data['URL'].apply(lambda x: format_url(x))
# split into equal size batches
batch_crit = pd.cut(data.index, num_batches, right=True).categories
batches = []
for i in batch_crit:
    start = i.left
    stop = i.right
    batches = batches + [data[(data.index > start) &
                             (data.index <= stop)]]
                                 
# Main loop function ##############################################################

def main():
    # for tracking progress
    completed = 0
    
    # for async multiprocessing
    manager = mp.Manager()
    q = manager.Queue()    
    pool = mp.Pool(mp.cpu_count() + 2)
    
    # for tracking progress
    very_start = time() 

    #fire off workers
    jobs = []
    for i in range(num_batches):
        jobs.append(pool.apply_async(worker, (i,)))

    # collect results from the workers through the pool result queue
    for job in jobs: 
        result = job.get() # once job (batch process) is done 
        
        # write text result to file
        for r in result.iterrows(): 
            writer.writerow(r[1]) # columns names
        # write to file on disk now, instead of later
        fo.flush() 
        os.fsync(fo) 
        
        # Update progress on console
        completed = completed + 1
        sys.stdout.write(f"\rCompleted Jobs: {completed}/{num_batches} (minutes: {np.round((time() - very_start)/60,2)})")
        sys.stdout.flush()
    
    # end the multiprocessing pool
    pool.close()
    pool.join()
    
    # close the output file we were writing to
    fo.close()
   
"""
This file does the same work, but not taking advantage of multi-processing. It was here just to compare
performance when developing. Multiprocessing has shown significant efficiencies over single and is highly 
recommended for scraping a large number of business URLs. It will save you time!
"""
def test_single_process():
    very_start = time()
    urls = data['URL']
    texts = []
    for i in range(len(data)):
        url = urls[i]
        # get text from domain and all chosen sub links
        ts, _, _ = scrape_page(url)
        # tsp: texts processed
        tsp = []
        # processes each page and concat into one long string
        for t in ts:
            # tp: text processed
            tp = process_text(t)
            tsp = tsp + [tp]
        # tl: text long
        if len(tsp) == 0:
            tl = ''
        else:
            tl = ' '.join(tsp)[:100] # can be very long, may want to reduce size in some intelligent way
        texts = texts + [tl]
        if i % 10 == 0:
            sys.stdout.write(f"\rCompleted URLs: {i}/{len(data)} (minutes: {np.round((time() - very_start)/60,2)})")
            sys.stdout.flush()
    data['Text'] = texts
    
    sys.stdout.write(f"Completed in {np.round((time() - very_start)/60,2)} minutes)")
    sys.stdout.flush()
    
    data.to_csv(file_out, index=False)
    
###############################################################

# Run ##############################################################

if __name__ == "__main__":
   main()
   

