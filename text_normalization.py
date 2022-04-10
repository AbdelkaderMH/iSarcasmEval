#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created by: Mohamed Salem Elhady  
Email: mohamed.elaraby@alumni.ubc.ca
Text Normalization: V1 
'''
import sys
import re
import emojis
from emoji import UNICODE_EMOJI
#sys.setdefaultencoding('utf-8')
##########################Clean Text Data #######################################
########################Global Variable Declaration##############################
list_seeds = ['سبحان الله', 'الله أكبر', 'اللهم', 'بسم الله', 'يا رب', 'العضيم', 'سبحان', 'يارب', 'قران', 'quran',
              'حديث', 'hadith', 'صلاه_الفجر', '﴾', 'ﷺ', 'صحيح البخاري', 'صحيح مسلم', 'يآرب', 'سورة']
MaxWordPerTweet=7
#################################################################################
def is_emoji(s):
    return s in UNICODE_EMOJI

# add space near your emoji
def add_space(text):
    return ''.join(' ' + char if is_emoji(char) else char for char in text).strip()

def clean(sent):
    """clean data from any English char, emoticons, underscore, and repeated > 2
    str -> str"""
    p1 = re.compile('\W')
    p2 = re.compile('\s+')
    sent = re.sub(r"http\S+", "", sent)
    sent = ReplaceThreeOrMore(sent)
    sent = remove_unicode_diac(sent)
    sent = sent.replace('_', ' ')
    sent = re.sub(r'[A-Za-z0-9]', r'', sent)
    sent = re.sub(p1, ' ', sent)
    sent = re.sub(p2, ' ', sent)
    return sent

def tokenize_emojis(tweet):
    return list(emojis.get(tweet))

def replace_emoji(sent):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'[MASK]', sent)

def preprocess(tweet):
    tweet = add_space(tweet)
    emos = tokenize_emojis(tweet)
    sent = remove_unicode_diac(tweet)
    sent = re.sub(r'(?:@[\w_]+)', "user", sent)
    sent = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', "url", sent)
    sent = sent.replace('_', ' ')
    sent = sent.replace('#', ' ')
    if len(emos) > 0:
        sent = sent + ' [SEP] '  + ' '.join(emos)
    #    #sent = sent + ' [SEP] ' + clean_unicode(tweet) + ' [SEP] ' + ' '.join(emos)

    #else:
    #    sent = sent + ' [SEP] ' + clean_unicode(tweet)
    return sent

def preprocess_last(tweet, k=0):
    emos = tokenize_emojis(tweet)
    sent = remove_unicode_diac(tweet)
    sent = re.sub(r'(?:@[\w_]+)', "", sent)
    sent = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', "", sent)
    sent = sent.replace('_', ' ')
    sent = sent.replace('#', ' ')
    if k == 0:
        sent = sent
    elif k ==1 :
        sent = sent + ' [SEP] ' + ' '.join(emos)
    elif k==2 :
        sent = sent + ' [SEP] ' + clean_unicode(tweet) + ' [SEP] ' + ' '.join(emos)
    elif k == 3:
        sent = sent + ' [SEP] ' + clean_unicode(tweet)
    else:
        sent = replace_emoji(sent)
        sent = sent + ' [SEP] ' + clean_unicode(tweet) + ' [SEP] ' + ' '.join(emos)

    return sent


def normalize(sent):
    """clean data from any English char, emoticons, underscore, and repeated > 2
    str -> str"""
    sent = re.sub(r'(?:@[\w_]+)', "user", sent)
    sent = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', "url", sent)
    #sent = re.sub(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", "hashtag", sent)
    sent = ReplaceThreeOrMore(sent)
    sent = remove_unicode_diac(sent)
    sent = sent.replace('_', ' ')
    return sent

def ReplaceThreeOrMore(s):
    # pattern to look for three or more repetitions of any character, including
    # newlines.
    pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
def norm_alif(text):
    text = text.replace(u"\u0625", u"\u0627")  # HAMZA below, with LETTER ALEF
    #text = text.replace(u"\u0621", u"\u0627")  # HAMZA, with LETTER ALEF
    text = text.replace(u"\u0622", u"\u0627")  # ALEF WITH MADDA ABOVE, with LETTER ALEF
    text = text.replace(u"\u0623", u"\u0627")  # ALEF WITH HAMZA ABOVE, with LETTER ALEF
    return text
def remove_unicode_diac(text):
    """Takes Arabic in utf-8 and returns same text without diac"""
    # Replace diacritics with nothing
    text = text.replace(u"\u064B", "")  # fatHatayn
    text = text.replace(u"\u064C", "")  # Dammatayn
    text = text.replace(u"\u064D", "")  # kasratayn
    text = text.replace(u"\u064E", "")  # fatHa
    text = text.replace(u"\u064F", "")  # Damma
    text = text.replace(u"\u0650", "")  # kasra
    text = text.replace(u"\u0651", "")  # shaddah
    text = text.replace(u"\u0652", "")  # sukuun
    text = text.replace(u"\u0670", "`")  # dagger 'alif
    return text
def norm_taa(text):
    text=text.replace(u"\u0629", u"\u0647") # taa' marbuuTa, with haa'
    #text=text.replace(u"\u064A", u"\u0649") # yaa' with 'alif maqSuura
    return text
def norm_yaa(text):
    if len(text)!=0:
        if text[-1] == u"\u064A":
            text = text[:-1] + text[-1].replace(u"\u064A", u"\u0649")  # yaa' with 'alif maqSuura
    return text

def NormForWord2Vec(text):
    text=norm_taa(text)
    text=norm_yaa(text)
    text=norm_alif(text)
    return text

def remove_nonunicode2(Tweet):
    ## defining set of unicode ##
    #u""
    #Tweet=Tweet.decode("utf-8")
    UniLex={ ## This is list of all arabic unicode characters in addition to space (to separate words)
            u"\u0622",
            u"\u0626",
            u"\u0628",
            u"\u062a",
            u"\u062c",
            u"\u06af",
            u"\u062e",
            u"\u0630",
            u"\u0632",
            u"\u0634",
            u"\u0636",
            u"\u0638",
            u"\u063a",
            u"\u0640",
            u"\u0642",
            u"\u0644",
            u"\u0646",
            u"\u0648",
            u"\u064a",
            u"\u0670",
            u"\u067e",
            u"\u0686",
            u"\u0621",
            u"\u0623",
            u"\u0625",
            u"\u06a4",
            u"\u0627",
            u"\u0629",
            u"\u062b",
            u"\u062d",
            u"\u062f",
            u"\u0631",
            u"\u0633",
            u"\u0635",
            u"\u0637",
            u"\u0639",
            u"\u0641",
            u"\u0643",
            u"\u0645",
            u"\u0647",
            u"\u0649",
            u"\u0671",
            ' ',
            '\n'
          }
    fin_tweet=""
    for c in Tweet:
        if c in UniLex:
           fin_tweet=fin_tweet+c
    return fin_tweet

###### Heuristics Calculations ######
def diac_counter(text):
    #text=text.decode("utf-8")
    diac = [u"\u064B",u"\u064C", u"\u064D", u"\u064E", u"\u064F", u"\u0650", u"\u0651", u"\u0652", u"\u0670"]
    diac_count=0
    for d in diac:
        diac_count+=text.count(d)
#         if d in text:
#             print(d)
#             diac_count+=1
    return diac_count
def check_seed(list_seeds, text):
    ""
    for word in list_seeds:
        text = text.lower()
        if word.decode("utf-8") in text:
            return True
    return False
def EnglishCount(text):
    printable = ['e', 'a', 'o', 't', 'i']
    count = 0
    for ch in printable:
        count += text.count(ch.lower())
    return count
########################################



def eliminate_single_char_words(Tweet):
    parts = Tweet.split(" ")
    cleaned_line_parts = []
    for P in parts:
        if len(P) != 1:
            cleaned_line_parts.append(P)
    cleaned_line = ' '.join(cleaned_line_parts)
    return cleaned_line
def clean_unicode(Tweet):
    tweet=normalize(Tweet.strip("\n"))
    if len(tweet) !=0:
        sentence = []
        for word in tweet.split(" "):
            word = remove_unicode_diac(word)
            word = norm_alif(word)
            word = norm_taa(word)
            word = norm_yaa(word)
            word = normalize(word)
            sentence.append(word)
        tweet = ' '.join(sentence)
        tweet =remove_nonunicode2(tweet)
        tweet =eliminate_single_char_words(tweet)
    return tweet

def clean_unicode2(Tweet):
    KeepUniOnly(Tweet)
    tweet=normalize(Tweet.strip("\n"))
    if len(tweet) !=0:
        sentence = []
        for word in tweet.split(" "):
            word = remove_unicode_diac(word)
            word = normalize(word)
            sentence.append(word)
        tweet = ' '.join(sentence)
        tweet =remove_nonunicode2(tweet)
        tweet =eliminate_single_char_words(tweet)
    return tweet

def NormCorpusFinal(Tweet):
    tweet=KeepUniOnly(Tweet)
    tweet=NormForWord2Vec(tweet)
    return tweet

def KeepUniOnly(Tweet):## this one is without normalization
    tweet=Tweet.replace("# "," ")
    tweet=tweet.replace("#"," ")
    tweet=tweet.replace("_"," ")
    tweet=tweet.replace(u"\u0657"," ")
    tweet=tweet.replace("\n"," ")
    tweet=remove_nonunicode2(tweet)
    tweet=eliminate_single_char_words(tweet)
    tweet=ReplaceThreeOrMore(tweet)
    return tweet

def get_charset(rawtext):
    chars = sorted(list(set(rawtext)))
    return chars

def DialectChecker(text):
    ##Based on Hueristics done by Hassan
    if (diac_counter(text)>5 or check_seed(list_seeds,text) or EnglishCount(text)>4 or "<URL>"  in text
        or text.count('#') >2 or '"'  in text or text.count('@') or "\"RT" in text or len(text.split(" ")) <7):
        return False
    else:
        return True

###############################################################
'''
Fread=open("Egypt_portion.txt",'r')
Fwriter=open("Egypt_portion_norm.txt",'w')
for line in Fread:
    cleaned_line=clean_unicode_for_w2v(line)
    Fwriter.write(str(cleaned_line))
Fwriter.close()
'''
