#!/usr/bin/env python

from __future__ import print_function
import sys
import os
import argparse
import tempfile


from KafNafParserPy import KafNafParser, KafNafParserMod
from collections import defaultdict
import KafNafParserPy

try:
    import pickle as pickler
except:
    import pickle as pickler


WORDNET_LEXICON_FILENAME = 'my_wn_exp_lex.bin'
TRAINING_FILENAME='training.expression'
TESTING_FILENAME='testing.expression'
RESOURCES_FOLDER = 'resources'
PARAMETERS_FILENAME = 'parameters.expression'

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
                
   
   

 
def extract_mpqa(naf_obj, list_token_ids, features, overall_options):
    mpqa_label = 'in_mpqa_lexicon'

    constituency_extractor = naf_obj.get_constituency_extractor()
    
    #if overall_options.get('use_mpqa_lexicon',False):
    #    mpqa_lexicon = overall_options.get('wordnet_lexicon')
    #else:
    #    #We dont extract anything
    #    return [mpqa_label]
    
    mpqa_lexicon = overall_options.get('mpqa_lexicon')
    if mpqa_lexicon is None:
        print('WARNING!! MPQA lexicon features selected by the lexicon has not been loaded!!!', file=sys.stderr)
        return [mpqa_label]
    
    for token_id in list_token_ids:
        term_id = naf_obj.termid_for_tokenid[token_id]
        term_obj = naf_obj.get_term(term_id)
        
        if features[token_id].get(mpqa_label) is None:
            if mpqa_lexicon is not None:
                #Wil be None if the lemma is not in the lexicon
                subjectivity_polarity = mpqa_lexicon.get_type_and_polarity(term_obj.get_lemma(),term_obj.get_pos())
            else:
                subjectivity_polarity = None
                
                
            if subjectivity_polarity is not None:
                features[token_id][mpqa_label] = '1'
                
                ##Add also the other in the same chunk
                if False and constituency_extractor is not None:
                    deepest_chunk_and_terms = constituency_extractor.get_deepest_phrase_for_termid(term_id)  #('NP', ['t6', 't7', 't8'])
                    for sub_term_id in deepest_chunk_and_terms[1]:
                        sub_token_ids = naf_obj.get_term(sub_term_id).get_span().get_span_ids()
                        for sub_token_id in sub_token_ids:
                            features[sub_token_id][mpqa_label] = '1'
            else:
                features[token_id][mpqa_label] = '0'
    return [mpqa_label]
        
        

def extract_wordnet_lexicon(naf_obj, list_token_ids, features, overall_parameters):
    this_label = 'in_wordnet_lexicon'
    
    if not overall_parameters.get('use_wordnet_lexicon', False):
        return [this_label]
    
    
    wordnet_lexicon = overall_parameters.get('wordnet_lexicon')
    if wordnet_lexicon is None:
        return [this_label]
    
    for token_id in list_token_ids:
        term_id = naf_obj.termid_for_tokenid[token_id]
        term_obj = naf_obj.get_term(term_id)
        frequency_in_wordnet_lexicon = 0
        if wordnet_lexicon is not None:
            frequency_in_lexicon = wordnet_lexicon.get_frequency(term_obj.get_lemma())
        
        if frequency_in_lexicon > 0:
            frequency_in_lexicon = 1
            
        features[token_id][this_label] = str(frequency_in_lexicon)
    return [this_label]
    

def extract_custom_lexicon(naf_obj,list_token_ids, features, custom_lexicon):
    this_label = 'in_custom_lexicon'
    for token_id in list_token_ids:
        first_token = naf_obj.get_token(token_id)
        text = first_token.get_text()
        this_polarity = custom_lexicon.get_polarity(text)
        if this_polarity is not None:
            features[token_id][this_label] = '1'
    return [this_label]
        

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


def extract_sentiment_nva(naf_obj,list_token_ids,features, overall_parameters):
    this_label = 'sentiment_nva'
    for token_id in list_token_ids:
        term_id = naf_obj.termid_for_tokenid[token_id]
        term_obj = naf_obj.get_term(term_id)
        if term_obj is not None:
            lemma = term_obj.get_lemma().lower()
            kaf_pos = term_obj.get_pos()
            normalised_pos = map_pos_to_sentiment_nva(kaf_pos)
            #lexicon_value = overall_parameters['sentiment-nva-gi42'].get((lemma,normalised_pos),None)
            lexicon_value = overall_parameters['sentiment-nva-gi42'].get(lemma,None)

            if lexicon_value is not None:
                #features[token_id][this_label] = lexicon_value
                features[token_id][this_label] = '1'
    return [this_label]

def extract_lexOut_90000(naf_obj,list_token_ids,features, overall_parameters):
    this_label = 'lexOut'
    for token_id in list_token_ids:
        term_id = naf_obj.termid_for_tokenid[token_id]
        term_obj = naf_obj.get_term(term_id)
        if term_obj is not None:
            lemma = term_obj.get_lemma().lower()
            kaf_pos = term_obj.get_pos()
            normalised_pos = map_pos_to_sentiment_nva(kaf_pos)
            #lexicon_value = overall_parameters['lexOut_90000_monovalue'].get((lemma,normalised_pos),None)
            lexicon_value = overall_parameters['lexOut_90000_monovalue'].get(lemma,None)

            if lexicon_value is not None:
                features[token_id][this_label] = lexicon_value
                #features[token_id][this_label] = '1'
    return [this_label]
        
        
def extract_from_lexicon(naf_obj,list_token_ids,features, overall_parameters):
    this_label = 'custom_lexicon'
    for token_id in list_token_ids:
        term_id = naf_obj.termid_for_tokenid[token_id]
        term_obj = naf_obj.get_term(term_id)
        if term_obj is not None:
            lemma = term_obj.get_lemma().lower()
            kaf_pos = term_obj.get_pos()
            lexicon_value = overall_parameters['custom_lexicon'].get_polarity(lemma)
            if lexicon_value is not None:
                features[token_id][this_label] = lexicon_value
                
    return [this_label]
            
    

def create_sequence(naf_obj, sentence_id, overall_parameters, list_opinions=[], output=sys.stdout, log=False):
    if log:
        print('\t\tCreating sequence for the sentence', sentence_id, 'and the opinions', ' '.join(opinion.get_id() for opinion in list_opinions), file=sys.stderr)
        
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
    
    
    #MPQA lexicon
    feature_labels = extract_mpqa(naf_obj, token_ids, features, overall_parameters)
    list_feature_labels.extend(feature_labels)
    
    #WORDNET LEXICON
    feature_labels = extract_wordnet_lexicon(naf_obj, token_ids, features, overall_parameters)
    list_feature_labels.extend(feature_labels)

    
    #Chunks
    feature_labels = extract_chunks(naf_obj, token_ids, features)
    list_feature_labels.extend(feature_labels)
    
    ##sentiment_vna
    #feature_labels = extract_sentiment_nva(naf_obj,token_ids,features, overall_parameters)
    #list_feature_labels.extend(feature_labels)
    
    ##lexOut 90000
    #feature_labels = extract_lexOut_90000(naf_obj,token_ids,features, overall_parameters)
    #list_feature_labels.extend(feature_labels)
    
    
    #USA STE BONICO
    ##This is good one to use
    #feature_labels = extract_from_lexicon(naf_obj,token_ids,features, overall_parameters)
    #list_feature_labels.extend(feature_labels)
    
    ########
    # We dont use the custom lexicon for now
    #feature_labels = extract_custom_lexicon(naf_obj, token_ids, features, overall_parameters.get('custom_lexicon'))
    #list_feature_labels.extend(feature_labels)
    ##############
    ####
    
    ##################
    ## THE TOKENS THAT ARE EXPRESSION
    opinion_expression_token_list = set()
    for opinion in list_opinions:
        opinion_expression_token_list = opinion_expression_token_list | set(get_token_ids_for_opinion_expression(naf_obj, opinion))
        
    ##PRINT THE SEQUENCE
    for token_id in token_ids:
        values_to_print = []
        values_to_print.append(naf_obj.filename+'#'+token_id)
        
        for feature_label in list_feature_labels:
            feature_value = features[token_id].get(feature_label,'-')
            if feature_value == None:
                feature_value = '-'
            
            feature_value = feature_value.replace(' ','_')
            values_to_print.append(feature_value)
            
        #######################################################
        #The class, in this case is the expression
        #######################################################
        this_class = None
        if token_id in opinion_expression_token_list:
            this_class = 'DSE'
        else:
            this_class = 'O'
        
        #In case we do not want to include the DSE label in the test file (it's not used by the system for tagging)    
        #if overall_parameters['is_test']:
        #    this_class = 'O'
             
        values_to_print.append(this_class)
        ############################################

        this_str = '\t'.join(values_to_print)
        output.write(this_str+'\n')        
        #print '\t'.join(values_to_print)
    output.write('\n')
    
    

def create_gold_standard(naf_obj,opinion_list,gold_fd):
    for opinion in opinion_list:
        opinion_expression_token_list = set(get_token_ids_for_opinion_expression(naf_obj, opinion))
        if len(opinion_expression_token_list) !=0:
            list_text_tokens = []
            for token_id in opinion_expression_token_list:
                token_obj = naf_obj.get_token(token_id)
                list_text_tokens.append((naf_obj.filename+'#'+token_id, token_obj.get_text(), int(token_obj.get_offset())))
            
            label = 'DSE'
            list_text_tokens.sort( key=lambda t: t[2])
            ids = [this_id for this_id, this_text, this_offset in list_text_tokens]
            values = [this_text for this_id, this_text, this_offset in list_text_tokens]
            gold_fd.write('%s\t%s\t%s\n' % (label,(' '.join(values)),' '.join(ids)))
            
           
def map_pos_to_sentiment_nva(this_pos):
    ##2930 a 105 adv 4180 n  1 POS 2022 v
    pos = 'X'
    if this_pos is not None:
        c = this_pos.lower()[0]
        if c in ['n','r']:
            pos = 'n'
        elif c == 'g':
            pos = 'a'
        elif c == 'a':
            pos ='adv'
        elif c == 'v':
            pos = 'v'
    return pos
         
def load_sentiment_nva_gi42():
    this_lexicon = {}
    path_to_file = '/home/izquierdo/cltl_repos/opinion_miner_deluxe/clean/lexicons/sentiment-nva-gi42.txt'
    fd = open(path_to_file)
    polarities_for_lemma = defaultdict(set)
    for line in fd:
        #zwoegen;v;negative
        #type of POS    2930 a 105 adv 4180 n  1 POS 2022 v
        fields = line.strip().split(';')
        #this_lexicon[(fields[0],fields[1])] = fields[2]
        polarities_for_lemma[fields[0]].add(fields[2])
        
    for lemma, polarities in list(polarities_for_lemma.items()):
        if len(polarities) == 1:
            this_lexicon[lemma] = list(polarities)[0]
    fd.close()
    return this_lexicon
            
def load_lexOut_90000():
    this_lexicon = {}
    path_to_file = '/home/izquierdo/cltl_repos/opinion_miner_deluxe/clean/lexicons/lexOut_90000_monovalue.txt'
    fd = open(path_to_file)
    polarities_for_lemma = defaultdict(set)
    for line in fd:
        #zwoegen;v;negative
        #type of POS    2930 a 105 adv 4180 n  1 POS 2022 v
        fields = line.strip().split('/')
        if len(fields) == 3:
            #this_lexicon[(fields[0],fields[1])] = fields[2]
            polarities_for_lemma[fields[0]].add(fields[2])
        
    for lemma, polarities in list(polarities_for_lemma.items()):
        if len(polarities) == 1:
            this_lexicon[lemma] = list(polarities)[0]
    fd.close()
    return this_lexicon

def main(inputfile, type, folder, overall_parameters={},log=False):
    files = []
    output_fd = None
    if type == 'train':
        if not os.path.isdir(folder):
            os.mkdir(folder)
        res_fol = os.path.join(folder,RESOURCES_FOLDER)
        if not os.path.isdir(res_fol):
            os.mkdir(res_fol)
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
        
    elif type == 'tag':
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
    elif type == 'test':
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
        
          
    ##Load the sentiment-nva-gi42.txt
    ##overall_parameters['sentiment-nva-gi42'] = load_sentiment_nva_gi42()  
    
    
    ##overall_parameters['lexOut_90000_monovalue'] = load_lexOut_90000()

    ###if overall_parameters['use_mpqa_lexicon']:
    from mpqa_lexicon import MPQA_subjectivity_lexicon
    overall_parameters['mpqa_lexicon'] = MPQA_subjectivity_lexicon()
    
    
    if overall_parameters.get('use_wordnet_lexicon', False):
        from wordnet_lexicon import WordnetLexicon
        wordnet_lexicon_expression = WordnetLexicon()
        complete_wn_filename = os.path.join(folder, RESOURCES_FOLDER, WORDNET_LEXICON_FILENAME) 

        if type == 'train':
            #We create it from the training files
            print('Creating WORDNET LEXICON FILE from %d files and storing it on %s' % (len(files), complete_wn_filename), file=sys.stderr)
            wordnet_lexicon_expression.create_from_files(files,'expression')
            wordnet_lexicon_expression.save_to_file(complete_wn_filename)
        else:
            #READ IT
            wordnet_lexicon_expression.load_from_file(complete_wn_filename)
        overall_parameters['wordnet_lexicon'] = wordnet_lexicon_expression
        
    gold_fd = None    
    gold_filename = overall_parameters.get('gold_standard')
    if gold_filename is not None:
        gold_fd = open(gold_filename ,'w')
          
    #Processing every file
    
    #### FOR THE CUSTOM LEXICON
    #from customized_lexicon import CustomizedLexicon
    #overall_parameters['custom_lexicon'] = CustomizedLexicon()
    #overall_parameters['custom_lexicon'].load_from_filename('EXP.nl')
    ###########################

    #from customized_lexicon import CustomizedLexicon
    #overall_parameters['custom_lexicon'] = CustomizedLexicon()
    #overall_parameters['custom_lexicon'].load_for_language('it')
    
    for filename in files:
        if log:
            print('EXPRESSION: processing file', filename, file=sys.stderr)
        
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
                    #if p.startswith('D-'):           
                    sentence_id = get_sentence_id_for_opinion(naf_obj,opinion)
                    if sentence_id is not None:
                        opinions_per_sentence[sentence_id].append(opinion)
                        num_opinions += 1
        if log:
            print('\tNum of opinions:', num_opinions, file=sys.stderr)
        
        
        if type == 'train':
            ############################
            # One sequence per sentence
            ############################
            for sentence_id in naf_obj.list_sentence_ids:
                opinions_in_sent = opinions_per_sentence.get(sentence_id,[])
                if len(opinions_in_sent) != 0:
                    ##Only sentences with opinions
                    create_sequence(naf_obj, sentence_id, overall_parameters, opinions_in_sent, output = output_fd)
        elif type == 'test':
            #TESTING CASE
            #For the testing, one sequence is created for every sentence
            for sentence_id in naf_obj.list_sentence_ids:
                opinions_in_sent = opinions_per_sentence.get(sentence_id,[])
                if len(opinions_in_sent) != 0:
                    #Only tested on sentences with opinions
                    create_sequence(naf_obj, sentence_id, overall_parameters, opinions_in_sent,output = output_fd)
                    
            ## Create the gold standard data also
            opinion_list = []
            for this_sentence, these_opinions in list(opinions_per_sentence.items()):
                opinion_list.extend(these_opinions)
            if gold_fd is not None:
                create_gold_standard(naf_obj,opinion_list,gold_fd)
        elif type == 'tag':
            #TAGGING CASE
            # All the sentences are considered
            for sentence_id in naf_obj.list_sentence_ids:
                create_sequence(naf_obj, sentence_id, overall_parameters, list_opinions = [],output = output_fd, log=log)
            
            
    if gold_fd is not None:
        gold_fd.close() 
        print('Gold standard in the file %s' % gold_fd.name, file=sys.stderr)
        
    output_fd.close()
    return output_fd.name
    


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description='Extract features and prepare for training/testing from a list of KAF/NAF files')
    argument_parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    argument_parser.add_argument('-i', dest='inputfile', required=True,help='Input file with a list of paths to KAF/NAF files (one per line)')
    argument_parser.add_argument('-t', dest='type', choices=['train', 'test','tag'], required=True,  default='train', help='Whether to train or test')
    argument_parser.add_argument('-mpqa', dest='use_mpqa_lexicon', action='store_true', help='Use the MPQA lexicon')
    argument_parser.add_argument('-wn_lex', dest='use_wn_lexicon', action='store_true', help='Use the WordNet lexicon')
    argument_parser.add_argument('-f', dest='folder', required=True, help='Folder to store the data')
    argument_parser.add_argument('-gs', dest='gold_standard', help='File to store the gold standard annotations (For evaluation)')
    args = argument_parser.parse_args()
    
    
    
    overall_parameters = {}
    
    if args.type == 'train':
        overall_parameters['use_mpqa_lexicon'] = args.use_mpqa_lexicon
        overall_parameters['use_wordnet_lexicon'] = args.use_wn_lexicon
        
    elif args.type == 'test':
        overall_parameters['is_test'] = True 
        overall_parameters['gold_standard'] = args.gold_standard
           
   
        
    main(args.inputfile,args.type, args.folder, overall_parameters)
    
