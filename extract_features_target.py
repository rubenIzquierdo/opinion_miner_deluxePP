#!/usr/bin/env python

from __future__ import print_function
import sys
import os
import tempfile
import argparse
from KafNafParserPy import KafNafParser
from collections import defaultdict

try:
    import pickle as pickler
except:
    import pickle as picker

TRAINING_FILENAME='training.target'
TESTING_FILENAME='testing.target'
PARAMETERS_FILENAME = 'parameters.target'

def create_structures(naf_obj, filename):
    '''
    Creates some structures and indexes that will be stored as public attributes of the kafnafparser object
    '''
    #naf_obj.filename = filename
    naf_obj.list_sentence_ids = []
    naf_obj.num_token_for_token_id = {}
    
    for num_token, token in enumerate(naf_obj.get_tokens()):
        sent_id = token.get_sent()
        if sent_id not in naf_obj.list_sentence_ids:
            naf_obj.list_sentence_ids.append(sent_id)
        naf_obj.num_token_for_token_id[token.get_id()] = num_token
    
    naf_obj.termid_for_tokenid = {}
    for term in naf_obj.get_terms():
        list_tokens= term.get_span().get_span_ids()
        for token_id in list_tokens:
            naf_obj.termid_for_tokenid[token_id] = term.get_id()
             

def get_sentence_id_for_opinion(naf_obj,this_opinion):
    '''
    Gets the sentence if for a given opinion (checks the opinon expression span)
    '''
    sent = None
    expression = this_opinion.get_expression()
    if expression is not None:
        span = expression.get_span()
        if span is not None:
            first_term_id = span.get_span_ids()[0]
            term_obj = naf_obj.get_term(first_term_id)
            first_token_id = term_obj.get_span().get_span_ids()[0]
            token_obj = naf_obj.get_token(first_token_id)
            sent = token_obj.get_sent()
    return sent


def get_token_ids_for_opinion_expression(naf_obj, opinion):
    '''
    Gets the list of token ids for the opinion expression
    '''
    tokens = []
    if isinstance(opinion,list):
        tokens = opinion
    else:
        expression = opinion.get_expression()
        if expression is not None:
            span = expression.get_span()
            if span is not None:
                expression_term_ids = span.get_span_ids()
            
                for term_id in expression_term_ids:
                    term_obj = naf_obj.get_term(term_id)
                    for token_id in term_obj.get_span().get_span_ids():
                        if token_id not in tokens:
                            tokens.append(token_id)
    return tokens
            

def get_token_ids_for_opinion_target(naf_obj, opinion):
    '''
    Gets the list of token ids for the opinion expression
    '''
    tokens = []
    target = opinion.get_target()
    if target is not None:
        span = target.get_span()
        if span is not None:
            target_term_ids = span.get_span_ids()
            
            for term_id in target_term_ids:
                term_obj = naf_obj.get_term(term_id)
                for token_id in term_obj.get_span().get_span_ids():
                    if token_id not in tokens:
                        tokens.append(token_id)
    return tokens


def extract_tokens(naf_obj, list_token_ids, features):
    '''
    Extract the token features for a list of token ids and stores the new features in a dictionary passed by reference
    '''
    this_label = 'token'
    for token_id in list_token_ids:
        token_obj = naf_obj.get_token(token_id)
        features[token_id][this_label] = token_obj.get_text()
    return [this_label]


def extract_terms_pos(naf_obj,list_token_ids, features):
    '''
    Extract the term and pos features for a list of token ids and stores the new features in a dictionary passed by reference
    '''
    lemma_label = 'lemma'
    pos_label = 'pos'
    
    
    for token_id in list_token_ids:
        term_id = naf_obj.termid_for_tokenid[token_id]
        term_obj = naf_obj.get_term(term_id)
        features[token_id][lemma_label] = term_obj.get_lemma()
        features[token_id][pos_label] = term_obj.get_pos()
    return [lemma_label, pos_label]
                


def extract_chunks(naf_obj, list_token_ids, features):
    this_label = 'deepest_chunk'
    extractor = naf_obj.get_constituency_extractor()
    if extractor is not None:
        for token_id in list_token_ids:
            term_id = naf_obj.termid_for_tokenid[token_id]
            term_obj = naf_obj.get_term(term_id)
            deepest_chunk_and_terms = extractor.get_deepest_phrase_for_termid(term_obj.get_id())
            features[token_id][this_label] = deepest_chunk_and_terms[0]
    return [this_label]


def extract_dse(naf_obj, token_ids, features, opinion):
    label = 'DSE'
    opinion_expression_token_list = get_token_ids_for_opinion_expression(naf_obj, opinion)
    if opinion_expression_token_list is not None:
        for token_id in token_ids:
            if token_id in opinion_expression_token_list:
                features[token_id][label] = 'DSE'
            else:
                features[token_id][label] = 'O'
    return [label]


                
def extract_distance_dse_target(naf_obj, token_ids, features, opinion):
    label = 'distance_to_dse'
    if opinion is not None:
        expression_ids = get_token_ids_for_opinion_expression(naf_obj, opinion)
        min_pos_eid = max_pos_eid = None
        for eid in expression_ids:
            this_pos = naf_obj.num_token_for_token_id[eid]
            if min_pos_eid is None or this_pos< min_pos_eid:
                min_pos_eid = this_pos
                
            if max_pos_eid is None or this_pos > max_pos_eid:
                max_pos_eid = this_pos
        ###
        for token_id in token_ids:
            position = naf_obj.num_token_for_token_id[token_id]
            if position>=min_pos_eid and position <= max_pos_eid:
                dist = 0
            else:
                #d1 = abs(position-min_pos_eid)
                #d2 = abs(position-max_pos_eid)
                d1 = position-min_pos_eid
                d2 = position-max_pos_eid
                dist = min(d1,d2)
            #features[token_id][label] = str(dist)
            features[token_id][label] = str(dist//3)
    return [label]



def extract_dependency_path_to_dse(naf_obj,token_ids,features, opinion):
    label = 'dependency_path'
    extractor = naf_obj.get_dependency_extractor()
    if extractor is not None:
        if isinstance(opinion,list):
            expression_term_ids = []
            for token_id in opinion:
                term_id = naf_obj.termid_for_tokenid[token_id]
                expression_term_ids.append(term_id)
        else:
            expression_term_ids = opinion.get_expression().get_span().get_span_ids()

        for token_id in token_ids:
            term_id = naf_obj.termid_for_tokenid[token_id]
            path = extractor.get_shortest_path_spans([term_id],expression_term_ids)
            if path is not None and len(path) > 0:
                features[token_id][label] = '#'.join(path)
    return [label]
    
    
def create_sequence(naf_obj, this_type, sentence_id, overall_parameters, opinion=None, output=sys.stdout, log=False):
    
    if log and opinion is not None:
        if isinstance(opinion, list):
            print('\t\tCreating sequence for the sentence', sentence_id, 'and the opinion with ids', opinion, file=sys.stderr)
        else:
            print('\t\tCreating sequence for the sentence', sentence_id, 'and the opinion', opinion.get_id(), file=sys.stderr)
            
    # Get all the token ids that belong to the sentence id
    token_ids = []
    features = {}
    for token in naf_obj.get_tokens():
        if token.get_sent() == sentence_id:
            token_ids.append(token.get_id())
            features[token.get_id()] = {}
    
    ####################################    
    ## EXTRACTING FEATURES
    ####################################
    list_feature_labels = []
    
    ## Tokens
    feature_labels = extract_tokens(naf_obj,token_ids, features)
    list_feature_labels.extend(feature_labels)
    
     ## Terms and POS
    feature_labels = extract_terms_pos(naf_obj,token_ids, features)
    list_feature_labels.extend(feature_labels)
    
    
    feature_labels = extract_distance_dse_target(naf_obj, token_ids, features, opinion)
    list_feature_labels.extend(feature_labels)
   
    
    feature_labels = extract_dependency_path_to_dse(naf_obj, token_ids, features, opinion)
    list_feature_labels.extend(feature_labels)
    
    
    #Chunks
    feature_labels = extract_chunks(naf_obj, token_ids, features)
    list_feature_labels.extend(feature_labels)
    
    
    
    feature_labels = extract_dse(naf_obj, token_ids, features, opinion)  
    list_feature_labels.extend(feature_labels)
    
    
    
      
    ##################
    ## THE TOKENS THAT ARE TARGETS
        
    opinion_target_token_list = set()
    if this_type == 'train' and opinion is not None:
        opinion_target_token_list = get_token_ids_for_opinion_target(naf_obj, opinion)
        
    ##PRINT THE SEQUENCE
    for token_id in token_ids:
        values_to_print = []
        values_to_print.append(naf_obj.filename+'#'+token_id)
        
        for feature_label in list_feature_labels:
            feature_value = features[token_id].get(feature_label,'-')
            if feature_value is None:
                feature_value = '-'
            feature_value = feature_value.replace(' ','_')
            values_to_print.append(feature_value)
            
        #######################################################
        #The class, in this case is the expression
        #######################################################
        this_class = None
        if token_id in opinion_target_token_list:
            this_class = 'TARGET'
        else:
            this_class = 'O'
        values_to_print.append(this_class)
        ############################################
        
        this_str = '\t'.join(values_to_print)
        output.write(this_str+'\n')        
        #print '\t'.join(values_to_print)
    output.write('\n')
    

def create_gold_standard_target(naf_obj,opinion_list,gold_fd):
    already_added = set()
    for opinion in opinion_list:
        opinion_target_token_list = set(get_token_ids_for_opinion_target(naf_obj, opinion))
        if len(opinion_target_token_list) != 0:
            list_text_tokens = []
            for token_id in opinion_target_token_list:
                token_obj = naf_obj.get_token(token_id)
                list_text_tokens.append((naf_obj.filename+'#'+token_id, token_obj.get_text(), int(token_obj.get_offset())))
            
            label = 'TARGET'
            list_text_tokens.sort( key=lambda t: t[2])
            ids = [this_id for this_id, this_text, this_offset in list_text_tokens]
            str_ids = ' '.join(ids)
            if str_ids not in already_added:
                values = [this_text for this_id, this_text, this_offset in list_text_tokens]
                gold_fd.write('%s\t%s\t%s\n' % (label,(' '.join(values)),str_ids))
                already_added.add(str_ids)
                
            
        

def main(inputfile, this_type, folder, overall_parameters = {}, detected_dse = {},log=False):
    files = []
    output_fd = None
    if this_type == 'train':
        output_fd = open(folder+'/'+TRAINING_FILENAME,'w')
            
        ##Save the parametes
        parameter_filename = os.path.join(folder,PARAMETERS_FILENAME)
        fd_parameter = open(parameter_filename,'w')
        pickler.dump(overall_parameters,fd_parameter,protocol=0)
        print('Parameters saved to file %s' % parameter_filename, file=sys.stderr)
        fd_parameter.close()
        
        #Input is a files with a list of files
        fin = open(inputfile,'r')
        for line in fin:
            files.append(line.strip())
        fin.close()
        
    elif this_type == 'tag':
        parameter_filename = os.path.join(folder,PARAMETERS_FILENAME)
        fd_param = open(parameter_filename,'rb')
        try:
            overall_parameters = pickler.load(fd_param,encoding='bytes')
        except TypeError:
            overall_parameters = pickler.load(fd_param)
        fd_param.close()

        #Input is a isngle file
        files.append(inputfile)
        
        #Output FD will be a temporary file
        output_fd = tempfile.NamedTemporaryFile('w', delete=False)
    elif this_type == 'test':
        parameter_filename = os.path.join(folder,PARAMETERS_FILENAME)
        fd_param = open(parameter_filename,'r')
        these_overall_parameters = pickler.load(fd_param)
        fd_param.close()
        for opt, val in list(these_overall_parameters.items()):
            overall_parameters[opt] = val
        
        #Input is a files with a list of files
        fin = open(inputfile,'r')
        for line in fin:
            files.append(line.strip())
        fin.close()
        output_fd = open(folder+'/'+TESTING_FILENAME,'w')
     
      
    gold_fd = None    
    gold_filename = overall_parameters.get('gold_standard')
    if gold_filename is not None:
        gold_fd = open(gold_filename ,'w')
          

    for filename in files:
        if log:
            print('TARGET: processing file', filename, file=sys.stderr)
        
        if isinstance(filename,KafNafParser):
            naf_obj = filename
        else:
            naf_obj = KafNafParser(filename)
            
        create_structures(naf_obj, filename)
        
        #Extract all the opinions
        opinions_per_sentence = defaultdict(list)
        num_opinions = 0
       
        
        for opinion in naf_obj.get_opinions():
            exp = opinion.get_expression()
            if exp is not None:
                p = exp.get_polarity()
                if p != 'NON-OPINIONATED':
                    target = opinion.get_target()
                    if target is not None:  
                        span = target.get_span()
                        if span is not None:
                            S = span.get_span_ids()
                            if len(S) != 0:    
                                sentence_id = get_sentence_id_for_opinion(naf_obj,opinion)
                                if sentence_id is not None:
                                    opinions_per_sentence[sentence_id].append(opinion)
                                    num_opinions += 1
                    
        if log:
            print('\tNum of opinions:', num_opinions, file=sys.stderr)
        
        if this_type == 'train':
            # For the train a sequence is created for every opinion
            #One sequence is created for every DSE (possible to have repeated sentences)
            sentences_with_opinions = set()
            for this_sentence, these_opinions in list(opinions_per_sentence.items()):
                for opinion in these_opinions:
                    sentences_with_opinions.add(this_sentence)
                    create_sequence(naf_obj, this_type, this_sentence, overall_parameters, opinion, output = output_fd)
            
            #Include the rest of sentence without opinions
            '''
            for sentence_id in naf_obj.list_sentence_ids:
                if sentence_id not in sentences_with_opinions:
                    create_sequence(naf_obj, sentence_id, overall_parameters, list_opinions=[])
            '''
                
        elif this_type=='tag':
            # Obtain the opinions per sentence per
            opinions_per_sentence = defaultdict(list)
            for list_name_ids, list_words in detected_dse:
                list_ids = [v[v.rfind('#')+1:] for v in list_name_ids]
                first_token = naf_obj.get_token(list_ids[0])
                sentence_for_opinion = first_token.get_sent()
                opinions_per_sentence[sentence_for_opinion].append(list_ids)
                
            for this_sentence, these_opinions in list(opinions_per_sentence.items()):
                for list_dse_token_ids in these_opinions:
                    create_sequence(naf_obj, this_type, this_sentence, overall_parameters, opinion = list_dse_token_ids, output = output_fd,log=log)  

        elif this_type=='test':
            #For the testing, one sequence is created for every sentence, with no opinion included
            opinion_list = []
            for this_sentence, these_opinions in list(opinions_per_sentence.items()):
                for opinion in these_opinions:
                    create_sequence(naf_obj, this_type, this_sentence, overall_parameters,opinion, output = output_fd)
                    opinion_list.append(opinion)
   
            if gold_fd is not None:
                create_gold_standard_target(naf_obj,opinion_list,gold_fd)
            
            
    if gold_fd is not None:
        gold_fd.close() 
        print('Gold standard in the file %s' % gold_fd.name, file=sys.stderr)
        
    return output_fd.name 
    

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description='Extract features and prepare for training/testing from a list of KAF/NAF files')
    argument_parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    argument_parser.add_argument('-i', dest='inputfile', required=True,help='Input file with a list of paths to KAF/NAF files (one per line)')
    argument_parser.add_argument('-t', dest='type', choices=['train', 'test','tag'], required=True,  default='train', help='Whether to train or test')
    argument_parser.add_argument('-f', dest='folder', required=True, help='Folder to store the data')
    argument_parser.add_argument('-gs', dest='gold_standard', help='File to store the gold standard annotations (For evaluation)')

    args = argument_parser.parse_args()

    overall_parameters = {}
    if args.type == 'test':
        overall_parameters['gold_standard'] = args.gold_standard

    detected_dse = [(['example_en.naf#w4'], ['nice']), (['example_en.naf#w9', 'example_en.naf#w10', 'example_en.naf#w11'], ['the', 'best', '!!'])]
       
    main(args.inputfile,args.type, args.folder, overall_parameters, detected_dse)
    
    
