"""
This code is adapted from the source code used in the paper
'Style Change Detection with Feed-forward Neural Networks (2019)'

Title: PAN_2019
Authors: Chaoyuan Zuo, Yu Zhao, and Ritwik Banerjee
Date: Jul 2, 2019
Availability: https://github.com/chzuo/PAN_2019
"""

import json
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import numpy as np
import pickle
import textstat
import time
from tqdm import tqdm
import os


def count_occurence(check_word_list, word_list_all):
    num_count = 0
    for w in check_word_list:
        if w in word_list_all:
            num_count += word_list_all[w]
    return num_count


def count_occurence_phrase(phrase_list, para):
    num_count = 0
    for phrase in phrase_list:
        num_count += para.count(phrase)
    return num_count


def extract_features(document):
    feature_all = []
    for para in document:

        sent_list = sent_tokenize(para)
        word_dict = {}

        sent_length_list = [0, 0, 0, 0, 0, 0]  # 0-10,10-20,20-30,30-40,40-50,>50
        pos_tag_list = [0] * 15
        for sent in sent_list:

            w_list = word_tokenize(sent)

            for (word, tag) in pos_tag(w_list):
                if tag in ['PRP']:
                    pos_tag_list[0] += 1
                if tag.startswith('J'):
                    pos_tag_list[1] += 1
                if tag.startswith('N'):
                    pos_tag_list[2] += 1
                if tag.startswith('V'):
                    pos_tag_list[3] += 1
                if tag in ['PRP', 'PRP$', 'WP', 'WP$']:
                    pos_tag_list[4] += 1
                elif tag in ['IN']:
                    pos_tag_list[5] += 1
                elif tag in ['CC']:
                    pos_tag_list[6] += 1
                elif tag in ['RB', 'RBR', 'RBS']:
                    pos_tag_list[7] += 1
                elif tag in ['DT', 'PDT', 'WDT']:
                    pos_tag_list[8] += 1
                elif tag in ['UH']:
                    pos_tag_list[9] += 1
                elif tag in ['MD']:
                    pos_tag_list[10] += 1
                if len(word) >= 8:
                    pos_tag_list[11] += 1
                elif len(word) in [2, 3, 4]:
                    pos_tag_list[12] += 1
                if word.isupper():
                    pos_tag_list[13] += 1
                elif word[0].isupper():
                    pos_tag_list[14] += 1

            num_words_sent = len(w_list)
            if num_words_sent >= 50:
                sent_length_list[-1] += 1
            else:
                sent_length_list[int(num_words_sent / 10)] += 1

            for w in w_list:
                if len(w) > 20:
                    w = '<Long_word>'
                word_dict.setdefault(w, 0)
                word_dict[w] += 1

        base_feat1 = [len(sent_list), len(word_dict)] + sent_length_list + pos_tag_list  # num_sentences, num_words

        special_char = [';', ':', '(', '/', '&', ')', '\\', '\'', '"', '%', '?', '!', '.', '*', '@']
        char_feat = [para.count(char) for char in special_char]

        with open('_function_words.json', 'r') as f:
            function_words = json.load(f)

        function_words_feature = []
        for w in function_words['words']:
            if w in word_dict:
                function_words_feature.append(word_dict[w])
            else:
                function_words_feature.append(0)

        function_phrase_feature = [para.count(p) for p in function_words['phrases']]

        with open('_difference_words.json', 'r') as f:
            difference_dict = json.load(f)

        difference_words_feat = [count_occurence(difference_dict['word']['number'][0], word_dict),
                                 count_occurence(difference_dict['word']['number'][1], word_dict),
                                 count_occurence(difference_dict['word']['spelling'][0], word_dict),
                                 count_occurence(difference_dict['word']['spelling'][1], word_dict),
                                 count_occurence_phrase(difference_dict['phrase'][0], para),
                                 count_occurence_phrase(difference_dict['phrase'][1], para)]

        textstat_feat = [textstat.flesch_reading_ease(para),
                         textstat.smog_index(para),
                         textstat.flesch_kincaid_grade(para),
                         textstat.coleman_liau_index(para),
                         textstat.automated_readability_index(para),
                         textstat.dale_chall_readability_score(para),
                         textstat.difficult_words(para),
                         textstat.linsear_write_formula(para),
                         textstat.gunning_fog(para)]

        feature = base_feat1 + function_words_feature + function_phrase_feature + difference_words_feat + char_feat + textstat_feat
        feature_all.append(feature)

    return np.asarray(feature_all)


def generate_features(documents):
    features_per_document = []
    features_per_paragraph = []

    with tqdm(documents, unit="document", desc=f"Generating features") as pbar:
        for doc in pbar:

            para_features = extract_features(doc)

            doc_features = sum(para_features)

            features_per_document.append(doc_features)
            features_per_paragraph.append(para_features)
    return np.array(features_per_document), np.array(features_per_paragraph, dtype=object)


def main():
    from utilities import load_documents

    # Load documents
    train_docs, train_doc_ids = load_documents('train')
    val_docs, val_doc_ids = load_documents('val')

    # NB! Generating features takes a long time
    train_doc_textf, train_par_textf = generate_features(train_docs)
    val_doc_textf, val_par_textf = generate_features(val_docs)

    timestring = time.strftime("%Y%m%d-%H%M")

    if not os.path.exists('./features'):
        os.makedirs('./features')

    with open('./features/' + timestring + '_doc_textf_train.pickle', 'wb') as handle:
        pickle.dump(train_doc_textf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./features/' + timestring + '_par_textf_train.pickle', 'wb') as handle:
        pickle.dump(train_par_textf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./features/' + timestring + '_doc_textf_val.pickle', 'wb') as handle:
        pickle.dump(val_doc_textf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./features/' + timestring + '_par_textf_val.pickle', 'wb') as handle:
        pickle.dump(val_par_textf, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
