#!/usr/bin/env python

from __future__ import print_function
from __future__ import print_function
import sys
from collections import defaultdict
from KafNafParserPy import KafNafParser


class Centity:
    def __init__(self, line=None):
        self.id = ''
        self.type = ''
        self.filename = ''
        self.word_list = []
        self.token_id_list = []
        self.filename = ''
        if line is not None:
            self.load_from_line(line)
            
    def create(self,this_id, this_type, this_filename, id_list, word_list):
        self.id = this_id
        self.type = this_type
        self.filename = this_filename
        self.word_list = word_list[:]
        self.token_id_list = id_list[:]


    def load_from_line(self,line):
        fields = line.strip().split('\t')
        self.type = fields[0]
        self.word_list = fields[1].split(' ')
        ids_with_filename = fields[2].split(' ')
        for id_with_filename in ids_with_filename:
            p = id_with_filename.rfind('#')
            self.filename = id_with_filename[:p]
            #self.token_id_list.append(id_with_filename)
            self.token_id_list.append(id_with_filename[p+1:])
    
    def to_line(self):
        #in the DS we need to include also the filename
        tokens_with_filename = [self.filename+'#'+token_id for token_id in self.token_id_list]
        line = '%s\t%s\t%s' % (self.type,' '.join(self.word_list),' '.join(tokens_with_filename))
        return line
    
    def __str__(self):
        s = ''
        s += 'Type: %s\n' % self.type
        s += 'Words: %s\n' % str(self.word_list)
        s += 'Filename: %s\n' % self.filename
        s += 'Ids:  %s\n' % str(self.token_id_list)
        return s
    
    def get_avg_position(self, naf_obj):
        offset_total = 0
        for token_id in self.token_id_list:
            token_obj = naf_obj.get_token(token_id)
            offset_total += int(token_obj.get_offset())
        avg_position =  1.0*offset_total/len(self.token_id_list)
        return avg_position
    
    def get_avg_position_num_tokens(self, naf_obj):
        list_ids_offset = []
        for token in naf_obj.get_tokens():
            list_ids_offset.append((token.get_id(),int(token.get_offset())))
        
        if hasattr(naf_obj, 'position_for_token'):
            pass
        else:
            naf_obj.position_for_token = {}
            numT = 0
            for token_id, token_offset in sorted(list_ids_offset, key=lambda t: -t[1]):
                naf_obj.position_for_token[token_id] = numT
                numT += 1    
            
        position_total = 0
        for token_id in self.token_id_list:
            position = naf_obj.position_for_token[token_id]
            position_total += position
            
        avg_position =  1.0*position_total/len(self.token_id_list)
        return avg_position
    
    def get_sentence(self, naf_obj):
        first_token = self.token_id_list[0]
        token_obj = naf_obj.get_token(first_token)
        sentence = token_obj.get_sent()
        return sentence

def load_entities(filename):
    list_entities = []
    fd = open(filename,'r')
    for line in fd:
        entity = Centity(line)
        list_entities.append(entity)
    fd.close()
    return list_entities
    

def match_entities(expression_entities, target_entities, knaf_obj):
    matched_pairs = []

    if len(expression_entities) > 0:
        for target in target_entities:
            target_sentence = target.get_sentence(knaf_obj)
            #position_for_target = target.get_avg_position(knaf_obj)
            position_for_target = target.get_avg_position_num_tokens(knaf_obj)
            
            expressions_with_distance = []
            #print 'Entity: ',expression.word_list, position_for_expression
            for expression in expression_entities:
                expression_sentence = expression.get_sentence(knaf_obj)
                if target_sentence == expression_sentence:
                    #position_for_expression = expression.get_avg_position(knaf_obj)
                    position_for_expression = expression.get_avg_position_num_tokens(knaf_obj)
                    distance = abs(position_for_expression-position_for_target)
                    expressions_with_distance.append((expression,distance))
                
            if len(expressions_with_distance) != 0:
                expressions_with_distance.sort(key=lambda t: t[1])
                #for target, d in expressions_with_distance:
                #    print '\t', target.word_list, d
                
                #We select the first one
                selected_expression = expressions_with_distance[0][0]
                #print 'FIXED:', target
                #for a,b in expressions_with_distance:
                #    print 'CANDIDATE', a.to_line(), b
                #print
                matched_pairs.append((selected_expression, target))
    return matched_pairs
 

    

if __name__ == '__main__':
    expression_filename = sys.argv[1] 	#test.mpqa.exp.csv
    target_filename = sys.argv[2]	#test.mpqa.tar.csv
    expression_entities = load_entities(expression_filename)
    target_entities = load_entities(target_filename)
    
    target_entities_per_filename = defaultdict(list)
    for t in target_entities:
        target_entities_per_filename[t.filename].append(t)
    
    
    for filename, list_targets in list(target_entities_per_filename.items()):
        knaf_obj = KafNafParser(filename)
        expression_candidates = []
        for expression in expression_entities:
            if expression.filename == filename:
                expression_candidates.append(expression)
        
        matched_pairs = match_entities(expression_candidates, list_targets, knaf_obj)
    
        for exp, tar in matched_pairs:
            print(exp.to_line())
            print(tar.to_line())
            print()
    

 
    
    
    
    
    
    
