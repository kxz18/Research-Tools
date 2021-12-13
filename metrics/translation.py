#!/usr/bin/python
# -*- coding:utf-8 -*-
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu


def tokenize_sent(sent):
    return sent.split(' ')


def meteor(references, hypothesis):
    '''token seperated by space, e.g.'''
    references = list(map(tokenize_sent, references))
    hypothesis = tokenize_sent(hypothesis)
    res = meteor_score(references, hypothesis)
    return res


def bleu(references, hypothesis):
    references = list(map(tokenize_sent, references))
    hypothesis = tokenize_sent(hypothesis)
    res = sentence_bleu(references, hypothesis)
    return res * 100
