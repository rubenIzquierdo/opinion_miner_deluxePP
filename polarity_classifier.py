from __future__ import print_function
from __future__ import print_function

import os
import pickle
import tempfile

from KafNafParserPy import KafNafParser
from collections import defaultdict
from subprocess import check_call


__here__ = os.path.realpath(os.path.dirname(__file__))

TRAIN_FILE='training_file.txt'
MODEL_FILE='model.bin'
INDEX_FILE='index_features.bin'

SVM_LEARN = os.path.join(__here__,'svm_light','svm_learn')
SVM_CLASSIFY = os.path.join(__here__,'svm_light','svm_classify')



class PolarityClassifier:
    def __init__(self, lang):
        self.index_features = {}    #Map string to int
        self.folder = None
        self.type_for_lemma = None
        self.__load_type_for_lemma(lang)
            
            
    def __load_type_for_lemma(self, lang):
        this_file = None
        
        if lang=='nl':
            this_file = os.path.join(__here__,'resources','lexicon.nl.txt')
        
        if this_file is not None:    
            self.type_for_lemma = {}
            fd = open(this_file)
            for line in fd:
                #tokens = line.strip().split(';')
                tokens = line.strip().split(';')
                self.type_for_lemma[tokens[0]] = tokens[2]
            fd.close()

    def extract_bow_tokens(self, this_obj, term_ids):
        features = []
        for tid in term_ids:
            term_obj = this_obj.get_term(tid)
            token_values = []
            for token_id in term_obj.get_span().get_span_ids():
                token_obj = this_obj.get_token(token_id)
                token_values.append(token_obj.get_text())
            features.append(('tokenBOW','_'.join(token_values)))
        return features
    
    def extract_bow_terms(self, this_obj, term_ids):
        features = []
        for tid in term_ids:
            term_obj = this_obj.get_term(tid)
            features.append(('termBOW',term_obj.get_lemma()))
        return features

    
    def extract_bigrams_tokens(self, this_obj, term_ids):
        features = []
        term_features = self.extract_bow_terms(this_obj, term_ids)
        if len(term_features) > 1:
            for idx in range(len(term_features)-1):
                bigram = term_features[idx][1]+'_'+term_features[idx+1][1]
                features.append(('BIG',bigram))
        return features

    def extract_trigrams_tokens(self, this_obj, term_ids):
        features = []
        term_features = self.extract_bow_terms(this_obj, term_ids)
        if len(term_features) >= 3:
            for idx in range(len(term_features)-2):
                bigram = term_features[idx][1]+'_'+term_features[idx+1][1]+'_'+term_features[idx+2][1]
                features.append(('TRIG',bigram))
        return features
    
            
    def extract_sentiment_templates(self, this_obj, term_ids):
        features = []
        if self.type_for_lemma is not None:
            lemmas = []
            for tid in term_ids:
                term_obj = this_obj.get_term(tid)
                lemmas.append(term_obj.get_lemma())
                
            template = []
            at_least_one = False
            for lemma in lemmas:
                if lemma in self.type_for_lemma:
                    at_least_one = True
                    template.append(self.type_for_lemma[lemma])
                else:
                    template.append('X')
                    
            if at_least_one:
                ### Bigrams
                if len(template) >= 2:
                    for idx in range(len(template)-1):
                        big = template[idx]+'_'+template[idx+1]
                        if 'POSITIVE' in big or 'NEGATIVE' in big:
                            features.append(('BIG_SENT', big))
                        
                ### Trigrams
                if len(template) >= 3:
                    for idx in range(len(template)-2):
                        trig = template[idx]+'_'+template[idx+1]+'_'+template[idx+2]
                        if 'POSITIVE' in trig or 'NEGATIVE' in trig:
                            features.append(('TRIG_SENT', trig))
                        
        return features   

    def extract_features(self, this_obj, list_term_ids):
        features = []
        
        bow_features = self.extract_bow_tokens(this_obj, list_term_ids)
        features.extend(bow_features)
        
        bow_terms = self.extract_bow_terms(this_obj, list_term_ids)
        features.extend(bow_terms)
        
        big_features = self.extract_bigrams_tokens(this_obj, list_term_ids)
        features.extend(big_features)
        
        trig_features = self.extract_trigrams_tokens(this_obj, list_term_ids)
        features.extend(trig_features)
        
        
        template_features = self.extract_sentiment_templates(this_obj, list_term_ids)
        features.extend(template_features)
        
        return features
    
    
    
    def write_example_to_file(self,fd,this_class,int_features):
        fd.write(this_class)
        features_sorted_by_index = sorted(list(int_features.items()), key=lambda t: t[0])
        for index_feat, freq_feat in features_sorted_by_index:
            fd.write(' %d:%d'% (index_feat, 1))
        fd.write('\n')
        
        

    def train(self,list_training_files, out_folder):
        self.folder= out_folder
        os.mkdir(self.folder)
        print('Creating output folder %s' % self.folder)
        
        training_fd = open(os.path.join(self.folder,TRAIN_FILE),'w')
        
        
        for this_file in list_training_files:
            print('\tEncoding training file %s' % this_file)
            
            this_obj = KafNafParser(this_file)
            num_pos = num_neg = 0
            for opinion in this_obj.get_opinions():
                opinion_expression = opinion.get_expression()
                polarity = opinion_expression.get_polarity()
                
                span_obj = opinion_expression.get_span()
                if span_obj is None:
                    continue
                
                list_term_ids = span_obj.get_span_ids()
                features = self.extract_features(this_obj, list_term_ids)
                
            
                int_features = self.encode_string_features(features, update_index=True) #Map feat index --> frequency
                
                if len(int_features) != 0:                
                    this_class = None
                    if self.is_positive(polarity):
                        this_class = '+1'
                        num_pos += 1
                    elif self.is_negative(polarity):
                        this_class = '-1'
                        num_neg += 1
                    
                    if this_class is not None:
                        self.write_example_to_file(training_fd, this_class, int_features)

            #END FOR
            print('\t\tNum positive examples: %d' % num_pos)
            print('\t\tNum negative examples: %d' % num_neg)
        training_fd.close()
        print('Training file at %s' % training_fd.name)
        
        ##RUN THE TRAINING
        training_cmd = [SVM_LEARN]
        
        training_cmd.append(training_fd.name)
        
        whole_model_file = os.path.join(self.folder, MODEL_FILE)
        training_cmd.append(whole_model_file)
        ret_code = check_call(training_cmd)
        print('Training done on %s with code %d' % (whole_model_file,ret_code))
        
        #Save also the index
        whole_index_file = os.path.join(self.folder,INDEX_FILE)
        index_fd = open(whole_index_file,'wb')
        pickle.dump(self.index_features, index_fd, -1)
        index_fd.close()
        print('Feature index saved to %s with %d features' % (whole_index_file,len(self.index_features)))
        
        
                
                
    def encode_string_features(self,string_features, update_index):
        int_features = defaultdict(int)     # Map of index --> frequency
        
        for type_feat, val_feat in string_features:
            whole_feature = '%s###%s' % (type_feat, val_feat)
            this_index = self.index_features.get(whole_feature)
            
            if this_index is not None:
                int_features[this_index] += 1
            else:
                if update_index:
                    this_index = len(self.index_features) + 1
                    self.index_features[whole_feature] = this_index
                    int_features[this_index] += 1
                else:
                    pass
                
        return int_features
                
    
    def is_positive(self, this_polarity):
        positive = False
        if this_polarity in ['Positive', 'StrongPositive']:
            positive = True
        elif 'polarity_dse=positive' in this_polarity:
            positive = True
        elif 'polarity_dse=uncertain-positive' in this_polarity:
            positive = True
        elif 'Positive' in this_polarity:
            positive = True
        
        return positive

    def is_negative(self, this_polarity):
        negative = False
        if this_polarity in ['Negative', 'StrongNegative']:
            negative = True
        elif 'polarity_dse=negative' in this_polarity:
            negative = True
        elif 'polarity_dse=uncertain-negative' in this_polarity:
            negative = True
        elif 'Negative' in this_polarity:
            positive = True
        
        return negative
        
    
    
    def load_models(self,folder):
        self.folder = folder
        
        #Load the index
        whole_index_file = os.path.join(self.folder,INDEX_FILE)        
        index_fd = open(whole_index_file,'rb')
        self.index_features = pickle.load(index_fd)
        index_fd.close()
        #print
        #print 'Feature index loaded from %s with %d features' % (whole_index_file,len(self.index_features))
        #print
        
        
    def decide_class(self,float_value):
        if float_value >= 0.0:
            return 'positive'
        else:
            return 'negative'
        
    def classify_list_opinions(self,this_obj, list_id_and_term_ids):
        class_for_opinion_id = {}
        features_for_opinion_id = {}
        
        test_fd = tempfile.NamedTemporaryFile(mode='w', delete=False)
        list_opinion_ids = []
        for opinion_id, term_ids in list_id_and_term_ids:
            string_features = self.extract_features(this_obj, term_ids)
            int_features = self.encode_string_features(string_features, update_index=False)
            self.write_example_to_file(test_fd, '0', int_features)
            list_opinion_ids.append(opinion_id)
            features_for_opinion_id[opinion_id] = string_features
        test_fd.close()
        
        output_filename = tempfile.mktemp()
        whole_model_file = os.path.join(self.folder, MODEL_FILE)
        #usage: svm_classify [options] example_file model_file output_file
        classify_cmd = [SVM_CLASSIFY]
        classify_cmd.append('-v')
        classify_cmd.append('0')
        classify_cmd.append(test_fd.name)
        classify_cmd.append(whole_model_file)
        classify_cmd.append(output_filename)
        
        ret_code = check_call(classify_cmd)
        
        fd_out = open(output_filename,'r')
        for num_line, line in enumerate(fd_out):
            value = float(line.strip())
            class_for_opinion_id[list_opinion_ids[num_line]] = self.decide_class(value)
        fd_out.close()
        
        #Remove the temp files
        os.remove(test_fd.name)
        os.remove(output_filename)
        
        return class_for_opinion_id, features_for_opinion_id
    
    
    def classify_kaf_naf_object(self,this_obj):
        list_ids_term_ids = []
        for opinion in this_obj.get_opinions():
            op_exp = opinion.get_expression()
            term_ids = op_exp.get_span().get_span_ids()
            list_ids_term_ids.append((opinion.get_id(),term_ids))
            
        class_for_opinion_id, features_for_opinion_id = self.classify_list_opinions(this_obj, list_ids_term_ids)
        
        #Modify the object
        for opinion in this_obj.get_opinions():
            opinion_id = opinion.get_id()
            this_polarity = class_for_opinion_id[opinion_id]
            op_exp = opinion.get_expression()
            op_exp.set_polarity(this_polarity)
            
        
        
        
    
            
