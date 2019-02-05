#!/usr/bin/env python

from __future__ import print_function
from __future__ import print_function
import os
import re

##from __init__ import PATH_MPQA_LEXICON

PATH_MPQA_LEXICON = './data/subjclueslen1-HLTEMNLP05.tff'

def normalize_pos(pos):
    pos = pos.lower()
    new_pos = pos
    if pos in ['adj','a'] or pos[0:2]=='jj':
        new_pos = 'a'
    elif pos in ['adverb','r'] or pos[0:2]=='rb':
        new_pos = 'r'
    elif pos in ['anypos']:
        new_pos = '*'
    elif pos in ['noun','n'] or pos[0:2]=='nn' or pos[0:2]=='np':
        new_pos = 'n'
    elif pos in ['verb','v'] or pos[0]=='v':
        new_pos = 'v'
    return new_pos


class MPQA_subjectivity_lexicon:
    def __init__(self):
        self.stemmed = {}
        self.stemmed_anypos = {}
        self.no_stemmed = {}
        self.no_stemmed_anypos = {}

        self.__load()
        
    def __load(self):
        # Format of lines: 
        # type=weaksubj len=1 word1=abandoned pos1=adj stemmed1=n priorpolarity=negative
        if os.path.exists(PATH_MPQA_LEXICON):
        
            fic = open(PATH_MPQA_LEXICON)
            for line in fic:
                line=line.strip()+' '
                this_type = re.findall('type=([^ ]+)', line)[0]
                word = re.findall('word1=([^ ]+)', line)[0]
                pos = re.findall('pos1=([^ ]+)', line)[0]
                stemmed = re.findall('stemmed1=([^ ]+)', line)[0]
                prior_polarity = re.findall('priorpolarity=([^ ]+)', line)[0]
                pos = normalize_pos(pos)
                if stemmed == 'y':
                    self.stemmed[(word,pos)] = (this_type,prior_polarity)
                    if True or pos == '*':  #anypos
                        self.stemmed_anypos[word] = (this_type,prior_polarity)

                elif stemmed == 'n':  
                    self.no_stemmed[(word,pos)] = (this_type,prior_polarity) 
                    if True or pos == '*':
                        self.no_stemmed_anypos[word] = (this_type,prior_polarity)

            fic.close()
        
    def print_all(self):
        for (word,pos), (this_type, this_polarity) in list(self.stemmed.items()):
            if this_polarity in ['positive','negative','neutral']:
                print('%s;%s;%s' % (word,pos,this_polarity.upper()))
        
    def get_type_and_polarity(self,word,pos=None):
        res = None
        if pos is not None:
            pos = normalize_pos(pos)
            
            # Try no stemmed with the given pos
            res = self.no_stemmed.get((word,pos))
            
            # Try stemmed with the given pos
            if res is None:
                res = self.stemmed.get((word,pos))
            
        # Try no stemmed with any pos    
        if res is None:
            res = self.no_stemmed_anypos.get(word)

        # Try stemm with any pos
        if res is None:
            res = self.stemmed_anypos.get(word)
            
            
        
        return res
            
            
if __name__ == '__main__':
    o = MPQA_subjectivity_lexicon()
    o.print_all()
    #print o.get_type_and_polarity('abidance','adj')
    
    
