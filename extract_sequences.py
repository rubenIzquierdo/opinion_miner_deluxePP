#!/usr/bin/env python


'''
Extract sequences from the ouput of CRF
'''
from __future__ import print_function
from __future__ import print_function

import sys


def extract_sequences(input, this_type):
    #Input can be a filename or a list of sentence
    my_input = None
    if isinstance(input, str):
        my_input = open(input,'r')
    elif isinstance(input,list):
        my_input = input
        
    current = []
    sequences_of_ids = []
    word_for_id = {}
    num_sequence = None
    for line in my_input:
        line = line.strip().decode()
        if line.startswith('#'):
            # # 1 0.025510
            fields = line.strip().split()
            num_sequence = int(fields[1])
        elif line == '':
            #breakline
            if len(current) != 0:
                sequences_of_ids.append((num_sequence, current))
                current = []
        else:
            #normal line
            fields = line.strip().split('\t')
            this_id = fields[0]
            this_class = fields[-1]
            token = fields[1]
            word_for_id[this_id] = token
            if this_class != 'O':
                 current.append(this_id)
            else:
                if len(current) != 0:
                    sequences_of_ids.append((num_sequence, current))
                    current = []
    if len(current) != 0:
        sequences_of_ids.append((num_sequence, current))
        current = []
    
    
    ##Remove those sequences that are completely contained in other
    indexes_to_remove = []
    for n1, (numseq1, s1) in enumerate(sequences_of_ids):
        #Should we remove s1?
        for n2, (numseq2, s2) in enumerate(sequences_of_ids):
            if n1 != n2 and numseq1 is not None and numseq2 is not None and int(numseq1) > int(numseq2):
                common = set(s1) & set(s2)
                if len(common) != 0:
                    indexes_to_remove.append(n1)
                    #print>>sys.stderr, 'Removed %s of sequence %d because overlaps with %s of seq %d' % (s1,numseq1, s2, numseq2)
                        
    #print 'Remove:'
    #for i in indexes_to_remove:
    #    print '   ',sequences_of_ids[i]
    
    these_sequences = []
    for n, (numseq, s) in enumerate(sequences_of_ids):
        if n not in indexes_to_remove:
            these_sequences.append(s)
    
    remove_duplicated = False
    already_printed = set()
    final_sequences = []
    for s in these_sequences:
        string_for_ids = ' '.join(s)
        if remove_duplicated:
            if string_for_ids in already_printed:
                continue
        already_printed.add(string_for_ids)
        words = [word_for_id[this_id] for this_id in s]
        ## REMOVE DUPLICATED???
        final_sequences.append((s, words))
        
    if isinstance(input, str):
        my_input.close()
        
    return final_sequences
        
if __name__ == '__main__':
    filename = sys.argv[1]
    this_type = sys.argv[2]
    sequences = extract_sequences(filename, this_type)
    for ids, words in sequences:
        print('%s\t%s\t%s' % (this_type,' '.join(words),' '.join(ids)))
        
    
