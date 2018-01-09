# Use fasttext to classify human rights tweets

import csv
import re
import random
import numpy as np
import pandas as pd
from pyfasttext import FastText
import matplotlib.pyplot as plt
import matplotlib.cm
from mpl_toolkits.basemap import Basemap

# get data into format fasttext accepts
HR_FILE = 'human_rights_training_sample_8-18-15.csv'
NON_HR_FILE = 'non_hr_training_sample_8-21-15.csv'
FAST_TRAIN_FILE = 'fasttrain.txt'
FAST_TEST_FILE = 'fasttest.txt'
FAST_MODEL = 'fastmodel'
UNLABELED_FILE = 'first200k.csv'

def clean_text(text):
    """Preprocesses text"""
    cleaned_text = text.lower().strip()
    # remove punctuation
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
    # remove URLs
    cleaned_text = re.sub(r'http\S+', '', cleaned_text)
    return cleaned_text

with open(HR_FILE, 'r') as hrfile, open(NON_HR_FILE, 'r') as nonhrfile:
    hr_reader = csv.reader(hrfile)
    nonhr_reader = csv.reader(nonhrfile)
    hr_header = next(hr_reader)
    hr_text_idx = hr_header.index('text')
    hr_lines = 0                # keep track of how many hr examples there are
    nonhr_lines = 0
    with open(FAST_TRAIN_FILE, 'w') as fasttrain, open(FAST_TEST_FILE, 'w') as fasttest:
        for line in hr_reader:
            hr_lines += 1
            cleaned_line = clean_text(line[hr_text_idx])
            towrite = '__label__hr {0} \n'.format(cleaned_line)
            if random.random() < 0.8:
                fasttrain.write(towrite)
            else:
                fasttest.write(towrite)
        for line in nonhr_reader:
            nonhr_lines += 1
            # 11th column has text
            cleaned_line = clean_text(line[11])
            towrite = '__label__nonhr {0} \n'.format(cleaned_line)
            if random.random() < 0.8:
                fasttrain.write(towrite)
            else:
                fasttest.write(towrite)



def confusion_matrix(fast_file, fasttext, label_dict):
    """Generate confusion matrix from test file.
    
    Args:
        fast_file (str): Path to file in fasttext format to generate confusion matrix of
        fasttext: Trained supervised fasttext model to be used for prediction
        label_dict (dict): Dict that maps labels to indices in confusion matrix

    Returns:
        A k*k numpy array that shows false and true positives and negatives for each class
        i.e. a confusion matrix, where k is the number of classes.
    """
    actual = []
    test_text = []
    with open(fast_file, 'r') as fastfile:
        for i, line in enumerate(fastfile):
            test_text_list = re.findall(r'__label__\w+ (.+)', line)
            label_list = re.findall(r'__label__(.\w+)', line)
            if label_list and test_text_list:
                if len(label_list) == 1 and len(test_text_list) == 1:
                    actual.append(label_list[0])
                    # newline for fasttext predict
                    test_text.append(test_text_list[0] + '\n')
                else:
                    # need to check what is going on here
                    pass
            else:
                pass

    # predicted labels
    predicted = fasttext.predict(test_text)
    # output is list of lists, flatten
    predicted = [label for label_list in predicted for label in label_list]
    assert len(actual) == len(predicted)

    conf_matrix = np.zeros([fasttext.nlabels, fasttext.nlabels])
    # use dictionary to map labels to numbers
    correct = 0
    for a, p in zip(actual, predicted):
        conf_matrix[label_dict[a]][label_dict[p]] += 1
    return conf_matrix

def precision_recall(conf_matrix, label, label_dict):
    """Calculates precision and recall from confusion matrix.

    Args:
        conf_matrix: Confusion matrix (numpy array)
        label: Class which you want to calculate precision and recall for
        label_dict: Dict that maps labels to indices in conf_matrix

    Returns:
        The precision and recall.
    """
    label_index = label_dict[label]
    precision = conf_matrix[label_index, label_index]/sum(conf_matrix[:, label_index])
    recall = conf_matrix[label_index, label_index]/sum(conf_matrix[label_index, :])
    return precision, recall

label_dict = {'hr': 1, 'nonhr': 0}
n_trials = 60                    # how many times to run random search

results = [{} for x in range(n_trials)]

def train_fasttext(input_file, output_model, params):
    """Train supervised fastText model"""
    fasttext = FastText()
    fasttext.supervised(input=input_file,
                        output=output_model,
                        **params)
    return fasttext

for i in range(n_trials):
    param_results = {'dim': random.randint(50, 200),
                     'epoch': random.randint(1, 50),
                     'lr': random.uniform(0.1, 1.0),
                     'minCount': random.randint(1, 10),
                     'wordNgrams': random.randint(1, 5)}
    print(param_results)
    # train fasttext model
    fasttext = train_fasttext(FAST_TRAIN_FILE, FAST_MODEL, param_results)

    # test
    # in 2 class problems, will just give accuracy
    fasttext.test(FAST_TEST_FILE)

    conf_matrix = confusion_matrix(FAST_TEST_FILE, fasttext, label_dict)
    precision, recall = precision_recall(conf_matrix, 'hr', label_dict)
    param_results['precision'] = precision

    results[i] = param_results


results = sorted(results, key=lambda x: x['precision'])
best_param = results[-1]
best_precision = best_param.pop('precision', None)

# retrain model
# model not saved to save on HD space
fasttext = train_fasttext(FAST_TRAIN_FILE, FAST_MODEL, best_param)
    
# label unlabeled set
cleanedtext_unlabeled = []
text_unlabeled = []
latlong_unlabeled = []
with open(UNLABELED_FILE, 'r') as csvfile:
    reader = csv.reader(csvfile)
    # text is in 10th column, longlat in 24th and 25th column
    for line in reader:
        if line[8] == 'en' and line[25] != 'NA': # only english tweets and geocoded
            try:
                # need to add \n for predict
                latlong_unlabeled.append([float(line[25]), float(line[26])])
                cleanedtext_unlabeled.append(clean_text(line[11])+'\n')
                text_unlabeled.append(line[11])
            except ValueError:
                pass
        
labels_unlabeled = fasttext.predict(cleanedtext_unlabeled)


hr_unlabeled = []
hr_lat = []
hr_long = []
nonhr_unlabeled = []
nonhr_lat = []
nonhr_long = []
for i, label in enumerate(labels_unlabeled):
    if label == ['hr']:
        hr_unlabeled.append(text_unlabeled[i])
        hr_lat.append(latlong_unlabeled[i][0])
        hr_long.append(latlong_unlabeled[i][1])
    else:
        nonhr_unlabeled.append(text_unlabeled[i])
        nonhr_lat.append(latlong_unlabeled[i][0])
        nonhr_long.append(latlong_unlabeled[i][1])

# see some examples
print('\n'.join(random.sample(hr_unlabeled, 10)))
print('\n'.join(random.sample(nonhr_unlabeled, 10)))


def draw_map(lat, longit, color):
    """Draws a map of the US with the provided coordinates"""
    fig, ax = plt.subplots(figsize=(10, 20))
    # keep quality at [c]rude while still drawing
    # us coords
    # westlimit=-125.86; southlimit=24.09; eastlimit=-63.89; northlimit=49.3
    m = Basemap(resolution='c',
                projection='merc',
                lat_0=37, lon_0=-90,
                llcrnrlon=-125.86, llcrnrlat=24.09,
                urcrnrlon=-63.89, urcrnrlat=49.3
    )

    m.drawmapboundary()
    m.fillcontinents()
    m.drawcoastlines()

    # plot human rights tweets position
    x, y = m(lat, longit)
    m.plot(x, y, 'o', markersize=1, color=color)

    m.readshapefile('usaShape/county/cb_2016_us_county_500k', 'county')

draw_map(hr_lat, hr_long, "red")
draw_map(nonhr_lat, nonhr_long, "blue")
