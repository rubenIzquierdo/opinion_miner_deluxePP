#!/usr/bin/env python

from __future__ import print_function
from __future__ import print_function
import sys

from polarity_classifier import PolarityClassifier
from KafNafParserPy import KafNafParser


if __name__ == '__main__':
    
    files = []
    fd = open('nl.list.test')
    for line in fd:
        files.append(line.strip())
    fd.close()

    my_polarity_classifier = PolarityClassifier('nl')
    my_polarity_classifier.load_models(sys.argv[1])

    OK = WR = 1
    for example_file in files:
        this_obj = KafNafParser(example_file)
        
        
        my_polarity_classifier.classify_kaf_naf_object(this_obj)
        this_obj.dump()

        break
    
        GOLD = {}
        list_ids_term_ids = []
        for opinion in this_obj.get_opinions():
            op_exp = opinion.get_expression()
            polarity = op_exp.get_polarity()
            term_ids = op_exp.get_span().get_span_ids()
            list_ids_term_ids.append((opinion.get_id(),term_ids))
            GOLD[opinion.get_id()] = polarity
    
        
                
    
        class_for_opinion_id, features_for_opinion_id = my_polarity_classifier.classify_list_opinions(this_obj, list_ids_term_ids)
        for oid, c in list(class_for_opinion_id.items()):
            #print '%s Gold:%s   System:%s'  % (oid,GOLD[oid],c)
            #print '\tFeatures:', features_for_opinion_id[oid]
            if c.lower() in GOLD[oid].lower():
                OK +=1
            else:
                WR += 1
                
        
        #print
        #print '#'*50
        
    print('Testing with %d files' % len(files))
    print('OK=%d' % OK)
    print('WR=%d' % WR)
    print('Acc: %.2f' % (OK*100.0/(OK+WR)))
