import re

import enchant
import language_check
import nltk
from nltk import word_tokenize
from nltk.corpus import words #, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, sent_tokenize
import numpy as np
import pandas as pd
from textstat.textstat import textstat
import transitions_and_trigrams as tnt

# store spell check object in a variable 
# for english spell checking it is sufficient to just store the Dict
# (in spanish it's more involved)
SpellCheck = enchant.Dict()

tokenizer = RegexpTokenizer(r'\w+')
wordset = set(words.words())
lmtzr = WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')
tool = language_check.LanguageTool('en-US')

relevant_trigrams_set = tnt.relevant_trigram_set
transitions_set = tnt.transitions_set

# word ranking from `word_rank.csv`
word_rank = pd.read_csv('word_rank1.csv') 
rank_token = word_rank[['rank', 'token']]

# regular expressions for stage identification
stage1a_re = (r'\b(no)\W+(DT)\W{6}\w+\W+(VB|VBG|VBD|VBZ|VBP|VBN|MD)\b')
stage1b_re = (r'\b(NN(S|P|PS)|PRP|VB(G|N)|MD)\W{6}(not)\W+(RB)\W+\w+\W+(VB'
              r'(G|N))\b')
stage1c_re = (r'\b(not)\W+(RB)\W{6}\w+\W+(VBD|VBZ|VBP|MD)\b')
stage2a_re = (r'\b((do)\W+\w+\W+(not|n\'t)\W+(RB)|(dont)\W{6}\w+\W+)\W{6}'
              r'\w+\W+(VBG|VBD|VBZ|VBN|MD)\b')
stage2b_re = (r'\b(he|she|it|him|her)\W+\w+\W{6}((do)\W+\w+\W+(not|n\'t)'
              r'\W+(RB)|(dont))\b')
stage2c_re = (r'\b(i|you|we|they)\W+\w+\W{6}((does)'
              r'\W+\w+\W{6}(not|n\'t)|doesnt)\b')
stage3a_re = (r'\b(d(o|oes|id)|ha(ve|s|d)|be|a(m|re)|is|w(as|ere))'
              r'\W+\w+\W{6}(((do)\W+\w+\W+(not|n\'t))\W+(RB)|(dont))\b')
stage3b_re = (r'\b(ha(ve|s|d)|be|a(m|re)|is|w(as|ere))'
              r'\W+\w+\W{6}(not|n\'t)\b')
stage3c_re = (r'\b(MD)\W+((do)\W+\w+\W+(not|n\'t)|dont|not|n\'t)\b')
stage4a_re = (r'\b(i|you|we|they)\W+\w+\W{6}((do|did)\W+\w+\W+'
              r'(not|n\'t)|dont|didnt)\W+(RB|VBP)\W+\w+\W+(VB)\b')
stage4b_re = (r'\b(i|you|we|they)\W+\w+\W{6}((did)\W+\w+\W+'
              r'(not|n\'t)|didnt)\W+\w+\W{6}\w+\W+(VBD)\b')
stage4c_re = (r'\b(he|she|it)\W+(\w+|NNP)\W{6}((does)\W+\w+\W{6}(not|n\'t)'
              r'|doesnt)\W+\w+\W{6}\w+\W+(VB|VBZ)\b')

words_not_found = []


def spell_checker(tokens_low):
    """ takes tokens and checks each of them for proper spelling 
        uses the `enchant` library """
    
    words = tokens_low

    correct_words = 0
    # incorrect_words = []
    for word in words:
        if SpellCheck.check(word) is True:
            correct_words += 1
        else:
            pass
            # incorrect_words.append(word)
    if len(words) > 1:
        score = correct_words / len(words)
    else:
        score = 0

    return round(score, 3)


def cttr(tokens):
    """ This version of ttr corrects for size. 
        I prefer to pass in the lowercased `tokens_low` """
    types = set(tokens)
    if len(tokens) != 0:
        cttr = len(types) / np.sqrt(2 * len(tokens))
    else:
        cttr = 0
    return round(cttr, 3)


def extract_features(essay):
    """ Pass in the text of an essay and a dictionary of different metrics. """

    # everything is stored in this dictionary
    feat_ext = {}

    # Without punctuation, not lowered
    tokens = tokenizer.tokenize(essay)
    sentences = len(sent_tokenize(essay))
    types = set(tokens)
    feat_ext['num_types'] = len(set(tokens))
    feat_ext['num_tokens'] = len(tokens)
         
    # With punctuation, not lowered
    tokens_punct = word_tokenize(essay)
    tagged = nltk.pos_tag(tokens_punct)
         
    # With punctuation, lowered
    essay_low = essay.strip().lower()
    tokens_low = word_tokenize(essay_low)
    tagged_low = nltk.pos_tag(tokens_low)
         
    # Without punctuation, lowered
    tokens_low_np = tokenizer.tokenize(essay_low)
    types_low_np = set(tokens_low_np)
    num_types_low_np = len(set(tokens_low_np))
    num_tokens_low_np = len(tokens_low_np)
    content_types = [w for w in types_low_np if w not in stopwords]
    function_tokens = [w for w in tokens_low_np if w in stopwords]

    # spelling
    feat_ext['spelling_perc'] = spell_checker(tokens_low)

    # Word frequency ranking extractors 
    # print(content_types)
    rankings=[]
        # [rankings.append(rank_token[rank_token.token == word]['rank'])[0]
        #  for word in content_types if not Error]
    for word in content_types:
        count = list(rank_token[rank_token.token == word]['rank'])
        try:
            rankings.append(count[0])
        except IndexError:
            # print('The following word wasn\'t found:', word)
            words_not_found.append(word)
    feat_ext['rank_total'] = sum(rankings)
    try:
        feat_ext['rank_avg'] = round(feat_ext['rank_total']/ len(rankings), 4)
    except ZeroDivisionError:
        feat_ext['rank_avg'] = 0

    # Length feature extractor
    len_words = []
    [len_words.append(len(word)) for word in tokens]
    
    feat_ext['avg_len_word'] = round(sum(len_words) / feat_ext['num_tokens'], 4)
         
    # Sentence density feature extractor
    try:
        feat_ext['sent_density'] = round(sentences / feat_ext['num_tokens'] * 100, 2)
    except ZeroDivisionError:
        feat_ext['sent_density'] = 0
         
    # Lexical diversity feature extractor
    # why are we using `low_nlp` here? 
    try:
        feat_ext['ttr'] = round(num_types_low_np / num_tokens_low_np * 100, 2)
    except ZeroDivisionError:
        feat_ext['ttr'] = 0
         
    # English words feature extractor
    english_types = []
    for word in types_low_np:
        if word in wordset:
            english_types.append(word)
    feat_ext['english_usage'] = len(english_types)
         
    # Percent of relevant trigrams in essay
    a, b = zip(*tagged)
    trigram_set = set(nltk.trigrams(b))
    found_trigrams = relevant_trigrams_set & trigram_set
    feat_ext['pct_rel_trigrams'] = round(len(found_trigrams) / len(relevant_trigrams_set) * 100, 2)
         
    found_transitions = transitions_set & types_low_np
    feat_ext['pct_transitions'] = round(len(found_transitions) / len(transitions_set), 4) 
         
    matches = tool.check(essay)
    feat_ext['grammar_chk'] = round(len(matches)/len(tokens), 5)
         
    rules =[]
    for match in matches:
        match_list = list(match)
        match_rule = match_list[4]
        rules.append(match_rule)
    for rule in set(rules):
        grammar_error = rule
         
    # determiners
    det = len(re.findall(r'\b(DT)\b', str(tagged), flags=re.I))
    feat_ext['determiners'] = round(det/len(tokens), 4)
         
    # n_lemma_types
    lemma_types_list = []
    for word in types_low_np:
        lemma_types = lmtzr.lemmatize(word)
        lemma_types_list.append(lemma_types)
        bigram_lemma_types = nltk.bigrams(lemma_types_list)
        trigram_lemma_types = nltk.trigrams(lemma_types_list)
    feat_ext['nlemma_types'] = len(lemma_types_list)
    feat_ext['n_bigram_lemma_types'] = len(list(bigram_lemma_types))
    feat_ext['n_trigram_lemma_types'] = len(list(trigram_lemma_types))
         
    # n_lemmas
    lemma_tokens_list = []
    for word in tokens_low_np:
        lemma_tokens = lmtzr.lemmatize(word)
        lemma_tokens_list.append(lemma_tokens)
        bigram_lemmas = nltk.ngrams(lemma_tokens_list,2)
        trigram_lemmas = nltk.ngrams(lemma_tokens_list,3)
    feat_ext['nlemmas'] = len(lemma_tokens_list)
    feat_ext['n_bigram_lemmas'] = len(list(bigram_lemmas))
    feat_ext['n_trigram_lemmas'] = len(list(trigram_lemmas))
         
    # ncontent_words
    content = [w for w in tokens_low_np if w not in stopwords]
    feat_ext['ncontent_words'] = len(content)


    feat_ext['cttr'] = cttr(tokens_low)
         
    # noun_ttr
    nouns = []
    for word, tag in tagged:
        if re.search(r'\b(NN(S|P|PS))\b', tag):
                    nouns.append(word)
    try:
        feat_ext['noun_ttr'] = round(len(set(nouns))/len(nouns),4)
    except ZeroDivisionError:
        feat_ext['noun_ttr'] = 0

    # function_ttr
    feat_ext['nfunction_words'] = len(function_tokens)
    try:
        feat_ext['function_ttr'] = round(len(set(function_tokens))/len(function_tokens),4)
    except ZeroDivisionError:
        feat_ext['function_ttr'] = 1

    # conjunctions
    conj = len(re.findall(r'\b(and|but)\W+(CC)\b', str(tagged), flags=re.I))
    feat_ext['conjunctions'] = round(conj / len(tokens), 5)
             
    # Readability measures
     
    feat_ext['fre'] = textstat.flesch_reading_ease(essay)
    feat_ext['fkg'] = textstat.flesch_kincaid_grade(essay)
    feat_ext['cli'] = textstat.coleman_liau_index(essay)
    feat_ext['ari'] = textstat.automated_readability_index(essay)
    feat_ext['dcrs'] = textstat.dale_chall_readability_score(essay)
    feat_ext['dw'] = textstat.difficult_words(essay)
    feat_ext['lwf'] = textstat.linsear_write_formula(essay)
    feat_ext['gf'] = textstat.gunning_fog(essay)
         
    # Stages of negation
    stage1a = len(re.findall(stage1a_re, str(tagged), flags=re.I))
    stage1b = len(re.findall(stage1b_re, str(tagged), flags=re.I))
    stage1c = len(re.findall(stage1c_re, str(tagged), flags=re.I))   
    stage2a = len(re.findall(stage2a_re, str(tagged), flags=re.I))
    stage2b = len(re.findall(stage2b_re, str(tagged), flags=re.I))
    stage2c = len(re.findall(stage2c_re, str(tagged), flags=re.I))
    stage3a = len(re.findall(stage3a_re, str(tagged), flags=re.I))
    stage3b = len(re.findall(stage3b_re, str(tagged), flags=re.I))
    stage3c = len(re.findall(stage3c_re, str(tagged), flags=re.I))
    stage4a = len(re.findall(stage4a_re, str(tagged), flags=re.I))
    stage4b = len(re.findall(stage4b_re, str(tagged), flags=re.I))
    stage4c = len(re.findall(stage4c_re, str(tagged), flags=re.I))
            
    stage1 = stage1a + stage1b + stage1c
    stage2 = stage2a + stage2b + stage2c
    stage3 = stage3a + stage3b + stage3c
    stage4 = stage4a + stage4b + stage4c
            
    neg_usage = stage1 + stage2 + stage3 + stage4
    # use display logic instead, if stage != 0 then fire
    try: 
        feat_ext['s1a'] = round(stage1a*100/neg_usage,2)
        feat_ext['s1b'] = round(stage1b*100/neg_usage,2)
        feat_ext['s1c'] = round(stage1c*100/neg_usage,2)
        feat_ext['s2a'] = round(stage2a*100/neg_usage,2)
        feat_ext['s2b'] = round(stage2b*100/neg_usage,2)
        feat_ext['s2c'] = round(stage2c*100/neg_usage,2)
        feat_ext['s3a'] = round(stage3a*100/neg_usage,2)
        feat_ext['s3b'] = round(stage3b*100/neg_usage,2)
        feat_ext['s3c'] = round(stage3c*100/neg_usage,2)
        feat_ext['s4a'] = round(stage4a*100/neg_usage,2)
        feat_ext['s4b'] = round(stage4b*100/neg_usage,2)
        feat_ext['s4c'] = round(stage4c*100/neg_usage,2)
        
        feat_ext['s1'] = feat_ext['s1a'] + feat_ext['s1b'] + feat_ext['s1c']
        feat_ext['s2'] = feat_ext['s2a'] + feat_ext['s2b'] + feat_ext['s2c']
        feat_ext['s3'] = feat_ext['s3a'] + feat_ext['s3b'] + feat_ext['s3c']
        feat_ext['s4'] = feat_ext['s4a'] + feat_ext['s4b'] + feat_ext['s4c']

    except ZeroDivisionError:
        feat_ext['s1a'] = 0 
        feat_ext['s1b'] = 0
        feat_ext['s1c'] = 0 
        feat_ext['s2a'] = 0
        feat_ext['s2b'] = 0 
        feat_ext['s2c'] = 0
        feat_ext['s3a'] = 0 
        feat_ext['s3b'] = 0 
        feat_ext['s3c'] = 0 
        feat_ext['s4a'] = 0 
        feat_ext['s4b'] = 0 
        feat_ext['s4c'] = 0
        feat_ext['s1'] = 0.0
        feat_ext['s2'] = 0.0
        feat_ext['s3'] = 0.0
        feat_ext['s4'] = 0.0

    # assert len(extracted_features) == len(headings)

    # used to return `feat_ext` but it seemed more elegant to
    # return feat_ext
    feat_ext = dict(sorted(feat_ext.items()))
    feature_set = list(feat_ext.values())
    # [print(k) for k, v in feat_ext.items()]
    return np.array(feature_set).reshape(1, -1)
