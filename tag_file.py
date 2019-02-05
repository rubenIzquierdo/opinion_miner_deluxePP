#!/usr/bin/env python


from __future__ import print_function
import os
import sys
import argparse


from extract_features_expression import main as expression_feature_extractor
from extract_features_target import main as target_feature_extractor
from extract_features_holder import main as holder_feature_extractor
from extract_sequences import extract_sequences
import match_entities_by_distance as entity_matcher
from subprocess import Popen, PIPE

from path_crf import PATH_TO_CRF_TEST
from KafNafParserPy import *
from polarity_classifier import PolarityClassifier


__desc = 'Opinion Miner Deluxe'
__last_edited = '7jan2016'
__version = '3.0'

__here__ = os.path.realpath(os.path.dirname(__file__))


def add_opinions(opinion_triples,kaf_naf_obj):
    term_id_for_token_id = {}
    for term in kaf_naf_obj.get_terms():
        for token_id in term.get_span().get_span_ids():
            term_id_for_token_id[token_id] = term.get_id()
            
    opinion_ids_used = set()
    for opinion in kaf_naf_obj.get_opinions():
        opinion_ids_used.add(opinion.get_id())

    #Adding linguistic processor
    my_lp = Clp()
    my_lp.set_name(__desc)
    my_lp.set_version(__last_edited+'_'+__version)
    my_lp.set_timestamp()   ##Set to the current date and time
    kaf_naf_obj.add_linguistic_processor('opinions',my_lp)
        
    num_opinion = 0 

    for E, T, H in opinion_triples:
        E_term_ids = [term_id_for_token_id[tokenid] for tokenid in E.token_id_list if tokenid in term_id_for_token_id]
        if T is None:
            T_term_ids = []
        else:
            T_term_ids = [term_id_for_token_id[tokenid] for tokenid in T.token_id_list if tokenid in term_id_for_token_id]
        
        if H is None:
            H_term_ids =[]
        else:
            H_term_ids = [term_id_for_token_id[tokenid] for tokenid in H.token_id_list if tokenid in term_id_for_token_id]
        
        new_id = None
        while True:
            new_id = 'o'+str(num_opinion+1)
            if new_id not in opinion_ids_used:
                opinion_ids_used.add(new_id)
                break
            else:
                num_opinion += 1
                
        new_opinion = Copinion(type=kaf_naf_obj.get_type())
        new_opinion.set_id(new_id)
        
        #Create the holder
        if len(H_term_ids) != 0:
            span_hol = Cspan()
            span_hol.create_from_ids(H_term_ids)
            my_hol = Cholder()
            my_hol.set_span(span_hol)
            hol_text = ' '.join(H.word_list)
            my_hol.set_comment(hol_text)  
            new_opinion.set_holder(my_hol)  
            
        #Creating target
        if len(T_term_ids) != 0:
            span_tar = Cspan()
            span_tar.create_from_ids(T_term_ids)
            my_tar = opinion_data.Ctarget()
            my_tar.set_span(span_tar)
            tar_text = ' '.join(T.word_list)
            my_tar.set_comment(tar_text)
            new_opinion.set_target(my_tar)
            #########################    

        ##Creating expression
        span_exp = Cspan()
        span_exp.create_from_ids(E_term_ids)
        my_exp = Cexpression()
        my_exp.set_span(span_exp)
        my_exp.set_polarity('DSE')
        #if include_polarity_strength:
        my_exp.set_strength("1")
        exp_text = ' '.join(E.word_list)
        my_exp.set_comment(exp_text)
        new_opinion.set_expression(my_exp)
        
        kaf_naf_obj.add_opinion(new_opinion)
        #########################
        
        
        
        

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detects opinions in KAF/NAF files', epilog='Example of use:  cat example.naf | %(prog)s -d hotel')
    parser.add_argument('--version', action='version', version='%(prog)s __version')
    
    #parser.add_argument('-f',dest='input_file', required=True,help='Input KAF/NAF file')
    input_group = parser.add_mutually_exclusive_group(required=True)
    
    input_group.add_argument('-d', dest='domain', help='Domain for the model (hotel,news)')
    input_group.add_argument('-f', dest='path_to_folder', help='Path to a folder containing the model')
    
    parser.add_argument('-log',dest='log',action='store_true',help='Show log information')
    parser.add_argument('-polarity', dest='polarity', action='store_true', help='Run the polarity (positive/negative) classifier too')
    parser.add_argument('-keep-opinions',dest='keep_opinions',action='store_true',help='Keep the opinions from the input (by default will be deleted)')
    
    if len(sys.argv) == 1:
        #To print by default the help, in case 
        sys.argv.append('-h')
 
    args = parser.parse_args(sys.argv[1:])
    
    
    if sys.stdin.isatty():
        print('Input stream required', file=sys.stderr)
        print('Example usage: cat my_file.naf | %s' % sys.argv[0], file=sys.stderr)
        parser.print_help(sys.stderr)
        sys.exit(-1)
    
    if args.log:
        print('Path to CRF TEST: %s' % PATH_TO_CRF_TEST, file=sys.stderr)                                                  
    kaf_naf_obj = KafNafParser(sys.stdin)
    
    if not args.keep_opinions:
        kaf_naf_obj.remove_opinion_layer()
    
    # We need to set this manually because the identifier for CRF will be the concatenation
    # of the filename and the token id, and it's a <File> object if we create the kaf_naf_obj
    # from a open stream
    kaf_naf_obj.filename = 'stdin'
    
   
    language = kaf_naf_obj.get_language()
    if args.log:
        print('Language in the file: %s' % language, file=sys.stderr)
        
    model_folder=None
    if args.domain:
        model_folder='models/models_%s_%s' % (args.domain,language)
    else:
        model_folder=args.path_to_folder
    
    if args.log:
        print('Model folder: %s' % model_folder, file=sys.stderr)
    
    if not os.path.exists(model_folder):
        print('There are no models for the domain %s' % args.domain, file=sys.stderr)
        print('    Model folder should be: %s' % model_folder, file=sys.stderr)
        sys.exit(-1)
    
    #########################################
    ########  BEGIN  EXPRESSION PART     ####
    #########################################
    
    # 1) CALL TO THE FEATURE EXTRACTOR FOR EXPRESSIONS
    #feature_file = expression_feature_extractor(inputfile,'tag', model_folder)
    feature_file = expression_feature_extractor(kaf_naf_obj,'tag', model_folder, log=args.log)

    
    # 2) CALL TO THE MODEL
    expression_tagger_cmd = []
    expression_tagger_cmd.append(PATH_TO_CRF_TEST)
    expression_tagger_cmd.append('-m')
    expression_tagger_cmd.append(model_folder+'/model.expression')
    expression_tagger_cmd.append(feature_file)
    
    expression_tagger = Popen(' '.join(expression_tagger_cmd), shell=True, stdout=PIPE, stderr=PIPE)
    expression_out, expression_error = expression_tagger.communicate()
    
    #This variable stores a list of lines with the CRF output 
    expression_out_lines = expression_out.splitlines()

    ######
    
      
    # The expression sequences detected:
    #[(['example_en.naf#w4'], ['nice']), 
    # (['example_en.naf#w9', 'example_en.naf#w10', 'example_en.naf#w11'], ['the', 'best', '!!'])]
    expression_sequences = extract_sequences(expression_out_lines,'DSE')
    #########################################
    ########  END  EXPRESSION PART     ####
    #########################################
 
 
    #########################################
    ########  BEGIN TARGET PART          ####
    #########################################
    
    
    target_features_file = target_feature_extractor(kaf_naf_obj,'tag',model_folder,detected_dse=expression_sequences, log=args.log)


    target_tagger_cmd = []
    target_tagger_cmd.append(PATH_TO_CRF_TEST)
    target_tagger_cmd.append('-m')
    target_tagger_cmd.append(model_folder+'/model.target')
    target_tagger_cmd.append(target_features_file)
                    
    target_tagger = Popen(' '.join(target_tagger_cmd), shell=True, stdout=PIPE, stderr=PIPE)
    target_out, target_error = target_tagger.communicate()
                            
    #This variable stores a list of lines with the CRF output 
    target_out_lines = target_out.splitlines()
    target_sequences = extract_sequences(target_out_lines,'TARGET')
    
    #########################################
    ########  END TARGET PART          ####
    #########################################

    
    #########################################
    ########  BEGIN HOLDER PART          ####
    #########################################
    
    holder_features_file = holder_feature_extractor(kaf_naf_obj,'tag', model_folder, detected_dse=expression_sequences, log=args.log)
    
    holder_tagger_cmd = []
    holder_tagger_cmd.append(PATH_TO_CRF_TEST)
    holder_tagger_cmd.append('-m')
    holder_tagger_cmd.append(model_folder+'/model.holder')
    holder_tagger_cmd.append(holder_features_file)
                    
    holder_tagger = Popen(' '.join(holder_tagger_cmd), shell=True, stdout=PIPE, stderr=PIPE)
    holder_out, holder_error = holder_tagger.communicate()
                            
    #This variable stores a list of lines with the CRF output 
    holder_out_lines = holder_out.splitlines()

    holder_sequences = extract_sequences(holder_out_lines,'HOLDER')
        
    #########################################
    ########  END HOLDER PART            ####
    #########################################
    
    
    #################################################
    ########  EXPRESSION/TARGET  PART            ####
    ################################################# 
    expression_entities = []
    num_exp = 0
    for list_ids, list_words in expression_sequences:
        exp_entity = entity_matcher.Centity()
        #ids contain filename
        ids_with_no_filename = []
        for this_id in list_ids:
            p = this_id.rfind('#')
            ids_with_no_filename.append(this_id[p+1:])
        filename = this_id[:p]

        exp_entity.create('exp#%d' % num_exp, 'DSE', filename, ids_with_no_filename, list_words)
        expression_entities.append(exp_entity)
        num_exp+=1
    

    target_entities = []
    num_tar = 0
    for list_ids, list_words in target_sequences:
        tar_entity = entity_matcher.Centity()
        #ids contain filename
        ids_with_no_filename = []
        for this_id in list_ids:
            p = this_id.rfind('#')
            ids_with_no_filename.append(this_id[p+1:])
        filename = this_id[:p]

        tar_entity.create('tar#%d' % num_tar, 'TARGET', filename, ids_with_no_filename, list_words)
        target_entities.append(tar_entity)
        num_tar+=1
     
    #We set target fixed and one expression is selected for every target   
    #matched_exp_tar = entity_matcher.match_entities(expression_entities,target_entities)
    
    #We set expressions fixed
    matched_tar_exp = entity_matcher.match_entities(target_entities, expression_entities, kaf_naf_obj)
    
    holder_entities = []
    num_hol = 0
    for list_ids, list_words in holder_sequences:
        hol_entity = entity_matcher.Centity()
        #ids contain filename
        ids_with_no_filename = []
        for this_id in list_ids:
            p = this_id.rfind('#')
            ids_with_no_filename.append(this_id[p+1:])
        filename = this_id[:p]

        hol_entity.create('hol#%d' % num_tar, 'HOLDER', filename, ids_with_no_filename, list_words)
        holder_entities.append(hol_entity)
        num_hol+=1
       
    matched_hol_exp = entity_matcher.match_entities(holder_entities, expression_entities, kaf_naf_obj)
    
    
    
    ###CREATE THE FINAL TRIPLES
    final_triples = []
    for expression in expression_entities:
        selected_target = None
        selected_holder = None
        for this_target, this_exp in matched_tar_exp:
            if expression.id == this_exp.id:
                selected_target = this_target
                break
        
        for this_holder, this_exp in matched_hol_exp:
            if expression.id == this_exp.id:
                selected_holder = this_holder
                break
            
        final_triples.append((expression, selected_target, selected_holder))
        
        
    if args.log:
        print('FOUND ENTITIES', file=sys.stderr)
        print('  Expressions', file=sys.stderr)
        for list_ids, list_words in expression_sequences:
            print('    ==>', ' '.join(list_words), str(list_ids), file=sys.stderr)
        print('  Targets', file=sys.stderr)
        for list_ids, list_words in target_sequences:
            print('    ==>', ' '.join(list_words), str(list_ids), file=sys.stderr)
        print('  Holders', file=sys.stderr)
        for list_ids, list_words in holder_sequences:
            print('    ==>', ' '.join(list_words), str(list_ids), file=sys.stderr)
        print(file=sys.stderr)
        print(file=sys.stderr)
        print('  Complete opinions', file=sys.stderr)
        for e, t, h in final_triples:
            print('    ==>', file=sys.stderr)
            print('      Expression:', e.to_line(), file=sys.stderr)
            if t is None:
                print('      Target: NONE', file=sys.stderr)
            else:
                print('      Target:', t.to_line(), file=sys.stderr)
            
            if h is None:
                print('      Holder: NONE', file=sys.stderr)
            else:
                print('      Holder:', h.to_line(), file=sys.stderr)
            
             
    
    #Remove feature_file feature_file
    #Remove also the target file  target_features_file
    
    os.remove(feature_file)
    os.remove(target_features_file)
    os.remove(holder_features_file)
    
    
    ## CREATE THE KAF/NAF OPINIONS
    add_opinions(final_triples,kaf_naf_obj)
    
    if args.polarity:
        my_polarity_classifier = PolarityClassifier(language)
        my_polarity_classifier.load_models(os.path.join(__here__,'polarity_models',language))
        my_polarity_classifier.classify_kaf_naf_object(kaf_naf_obj)
    
    kaf_naf_obj.dump()
    
