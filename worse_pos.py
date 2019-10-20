#PART 1 - CORPUS
import random
import time
import csv
from collections import Counter
t1=time.time()
print("Warning: This program is long, and takes some time to execute, because of the big file sizes.")
print("It took around 30s on an i7 7700HQ laptop with 16 GB of RAM. Performance might vary.")
def combine_lists(l1, l2): 
    return list(map(lambda x, y:"{} {}".format(x,y), l1, l2)) 

def givetaggivenword(some_dict):
    temp_dict={}
    temp=list(some_dict.values())
    for a_dict in temp:
        for tag in a_dict:
            if tag in temp_dict:
                temp_dict[tag]=temp_dict[tag]+a_dict[tag]
            else:
                temp_dict[tag]=a_dict[tag]
    best_tag=keywithmaxval(temp_dict)
    return(best_tag)

#Function to return the key having maximum value in a dictionary
def keywithmaxval(dic):
     val=list(dic.values())
     key=list(dic.keys())
     return key[val.index(max(val))]

def givesingletons(arr): 
    freq = Counter(arr) 
    return [elem for elem in arr if freq[elem] == 1] 

#MODEL SCRAPPED AS NOT ENOUGH TIME
#Function to give a tag which is calculated randomly by using the test tag set frequency as weights
# def weighted_random_tag(tag_list):
#     import numpy as np
#     unique_elements, counts_elements = np.unique(tag_list, return_counts=True)
#     counts_elements=counts_elements/np.sum(counts_elements)
#     weighted_tag=np.random.choice(unique_elements,p=counts_elements)
#     return(weighted_tag)

#Open File and read brown.txt
file=open("brown.txt","r")
all_text=file.read()
file.close()
clean_text=all_text.strip()

#Get all words along with their tags
trainfile=open("brown-train.txt","w")
testfile=open("brown-test.txt","w")
all_words=clean_text.split()
all_sent=[]
sent=""

#Join words to form sentences using the following loop
i=0   #Number of sentences
for word in all_words:
    if word[-2:]=="/.":
        sent=sent+word+" "
        all_sent.append(sent.strip())
        if len(all_sent[-1])==3:        #This is to remove all duplicates
            # print("All sent of -2 is{}".format(all_sent[-2]))
            # print("All sent of -1 is{}".format(all_sent[-1]))
            # print("Current sent is{}".format(sent))
            # print(all_sent[-1])
            del all_sent[-1]
            i=i-1
        sent=""
        i=i+1
        continue
    sent=sent+word+" "

#The first 2000 sentences of all sentences will form training set, while remaining will form test dataset
train_sent=all_sent[:2000]
test_sent=all_sent[2000:]

trainfile.write('\n'.join(train_sent))
testfile.write('\n'.join(test_sent))
#Write these training and test datasets to files
trainfile.close()
testfile.close()
print("brown-train.txt saved succesfully.")
print("brown-test.txt saved succesfully.")


#PART 2 - TAGGER IMPLEMENTATION

#Subpart 1 - Unigram
print("------------UNIGRAM------------")
#Create a nested dictionary of form {WORD1:{Tag1:Freqeuncy,Tag2:Freqeuncy,Tag3:Freqeuncy...},WORD2:{Tag1:Freqeuncy,Tag2:Freqeuncy,Tag3:Freqeuncy...},WORD3:{Tag1:Freqeuncy,Tag2:Freqeuncy,Tag3:Freqeuncy...}...}
unigram_tagger_dict={}   #Nested Dictionary 
unigram_list=[]   #List of all unigrams
tag_list=[]     #List of all tags
for sent in train_sent:
    for word in sent.split():
        unigram=word.rsplit("/",1)[0]
        tag=word.rsplit("/",1)[1]
        unigram_list.append(unigram)
        tag_list.append(tag)
        #A Tag Dictionary for the current word i.e for current word {Tag1,Tag2,Tag3...}
        if unigram in unigram_tagger_dict:
            tag_dict=unigram_tagger_dict[unigram]
        else:
            tag_dict={}
        if tag not in tag_dict:
            tag_dict[tag]=0
        tag_dict[tag]=tag_dict[tag]+1
        unigram_tagger_dict[unigram]=tag_dict

#Get the list of all unique unigrams and tags
unigram_set=list(set(unigram_list))
tag_set=list(set(tag_list))

max_tag_unigram_dict={}
unigramfile=open("unigram-tag.txt","w")
#Find out the most frequent tag for each word in training set and store as a dictionary
for unigram in unigram_set:
    current_unigram_dict=unigram_tagger_dict[unigram]
    unigram_values=list(current_unigram_dict.values())
    unigram_keys=list(current_unigram_dict.keys())
    max_tag=unigram_keys[unigram_values.index(max(unigram_values))]
    max_tag_unigram_dict[unigram]=max_tag

#Write the dictionary to a file outside the loop to save time
unigramfile.write(str(max_tag_unigram_dict))
unigramfile.close()
print("unigram-tag.txt saved succesfully.")

#Assign the most frequent tag calculated above to all words in training set
unigramresultfile=open("unigram-results.txt","w")
unigramresult=""    #String that holds all sentences after they've been tagged using unigram model
true_unigam_tag_counts=0    #To count how many assigned tags match the original correct tags
false_unigam_tag_counts=0   #To count how many assigned tags were assigned wrongly
unknown_correct=0
all_unknown={}  #Dictionary of all unknown unigrams
unigram_confusion={}        # { (tag1(true), tag2(model)) : freq }
hapax=givesingletons(unigram_list)
hapax_tags=[]
for elem in hapax:
    hapax_tags.append(max_tag_unigram_dict[elem])
#We have multiple models to assign tags to unknown words
print("Enter model number you would like to use : 0,1 or 2 based on:")
print("Approach 0: Mark all unknowns as UNK tags")
print("Approach 1: For unknown unigrams, give them a random tag with equal prob (1/n)")
print("Approach 2: For unknown unigrams, give them a random tag where the random prob is based ONLY ON THE UNIGRAMS WHICH APPEARED ONCE in the training data set.")
inp=int(input("Enter your choice:\n"))
for sent in test_sent:
    for word in sent.split():
        #Extract unigram and true_tag from "unigram/true_tag"
        unigram=word.rsplit("/",1)[0]
        true_tag=word.rsplit("/",1)[1]
        #Find out tag based on our model:
        #If the current unigram is a known unigram, then assign it the tag calculated earlier
        if unigram in max_tag_unigram_dict:
            model_tag=max_tag_unigram_dict[unigram]
        #If it's unknown, we have various strategies for that
        else:
            if inp==0:
                model_tag="UNK"
            if inp==1:
                model_tag=random.choice(tag_set)
            # if inp==2:               #MODEL SCRAPPED AS NOT ENOUGH TIME
            #     model_tag=weighted_random_tag(tag_list)
            if inp==2:
                model_tag=random.choice(hapax_tags)
            if model_tag==true_tag:
                unknown_correct+=1
            all_unknown.setdefault(unigram,0)
            all_unknown[unigram]=all_unknown[unigram]+1
        unigramresult=unigramresult+"{}/{} ".format(unigram,model_tag)
        #Update true and false tag counters
        if true_tag==model_tag:
            true_unigam_tag_counts+=1
        else:
            false_unigam_tag_counts+=1
            #CONFUSION
            unigram_confusion.setdefault((true_tag,model_tag),0)
            unigram_confusion[(true_tag,model_tag)]+=1
    unigramresult=unigramresult+"\n"

unigramresultfile.write(unigramresult)
unigramresultfile.close()
print("unigram-results.txt saved succesfully.")

unigram_accuracy=100*true_unigam_tag_counts/(false_unigam_tag_counts+true_unigam_tag_counts)
unknown_accuracy=unknown_correct/len(all_unknown)
print("Unigram Tagger Accuracy is {}%".format(unigram_accuracy))
print("Total unknowns is {}".format(len(all_unknown)))
print("Unknown Accuracy is {}%".format(unknown_accuracy))
#all_unknown_list=list(all_unknown.keys())

#Subpart 2 - Bigram
print("------------BIGRAM------------")
next_word_list=all_words[1:]
bigram_word_list=combine_lists(all_words,next_word_list)

bigram_tagger_dict={}   # Word1:{Tag1:{Possible Next Tags: Count},Tag2:{Possible Next Tags: Count}},Word2:...
bigramfile=open("bigram-tag.txt","w")
bigramtagtext="The format is:\nCurrent Word:\n\tPrevious Tag:\n\t\tNext Tag :\tFrequency\n-------------------\n"
#Order is Count(previous,next)
for i in range(len(bigram_word_list)):
    bigram_4_parts=bigram_word_list[i].replace(" ","/").rsplit("/")
    prev_tag=bigram_4_parts[1]
    next_tag=bigram_4_parts[3]
    next_word=bigram_4_parts[2]
    if next_word in bigram_tagger_dict:
        next_word_dict=bigram_tagger_dict[next_word]
    else:
        next_word_dict={}
    both_tags=bigram_4_parts[1]+bigram_4_parts[3]
    if prev_tag in next_word_dict:
        tag_dict=next_word_dict[prev_tag]
    else:
        tag_dict={}
    if next_tag not in tag_dict:
        tag_dict[next_tag]=0
    tag_dict[next_tag]=tag_dict[next_tag]+1
    next_word_dict[prev_tag]=tag_dict
    bigram_tagger_dict[next_word]=next_word_dict

bigramfile.write(str(bigram_tagger_dict))
bigramfile.close()
print("bigram-tag.txt saved succesfully.")

#Calculate the most probable next tag given previous tag for current word:
bigramresultfile=open("bigram-results.txt","w")
bigramresult=""    #String that holds all sentences after they've been tagged using unigram model
true_bigam_tag_counts=0    #To count how many assigned tags match the original correct tags
false_bigam_tag_counts=0   #To count how many assigned tags were assigned wrongly
unknown_correct_bigram=0
all_unknown_bigram={}
bigram_confusion={}        # { (tag1(true), tag2(model)) : freq }
i=0
j=0
print("Enter model number you would like to use : 1 or 2 based on:")
print("Approach 1: For unknown words, give them a random tag with equal prob (1/n)")
print("Approach 2: For unknown words, give them a random tag where the random prob is based ONLY ON THE UNIGRAMS WHICH APPEARED ONCE in the training data set.")
inp2=int(input("Enter your choice:\n"))
starting_tag="."        #Because this is a new sentence.
for sent in test_sent:
    for word in sent.split():
        if i==0 and j==0:
            prev_tag=starting_tag
        #Extract unigram and true_tag from "unigram/true_tag"
        unigram=word.rsplit("/",1)[0]
        true_tag=word.rsplit("/",1)[1]
        if unigram in bigram_tagger_dict:
            try:
                bigram_model_tag=keywithmaxval(bigram_tagger_dict[unigram][prev_tag])
            except Exception as e:
                #WORD FOUND, BUT NO TAG FOR PREV_TAG FOR THIS WORD Unknown Model
                    if inp2==1:
                        bigram_model_tag=random.choice(tag_set)
                    if inp2==2:
                        bigram_model_tag=random.choice(hapax_tags)
                #bigram_model_tag=givetaggivenword(bigram_tagger_dict[unigram])
        else:
            #WORD NOT FOUND: Unkown Model
            if inp2==1:
                bigram_model_tag=random.choice(tag_set)
            if inp2==2:
                bigram_model_tag=random.choice(hapax_tags)
            all_unknown_bigram.setdefault(prev_tag,0)
            all_unknown_bigram[prev_tag]=all_unknown_bigram[prev_tag]+1
        bigramresult=bigramresult+"{}/{} ".format(unigram,bigram_model_tag)
        if true_tag==bigram_model_tag:
            true_bigam_tag_counts+=1
        else:
            false_bigam_tag_counts+=1
            #CONFUSION
            bigram_confusion.setdefault((true_tag,model_tag),0)
            bigram_confusion[(true_tag,model_tag)]+=1
        prev_tag=bigram_model_tag
        j+=1
    bigramresult=bigramresult+"\n"
    i+=1
bigramresultfile.write(bigramresult)
bigramresultfile.close()
print("bigram-results.txt saved succesfully.")

bigram_accuracy=100*true_bigam_tag_counts/(false_bigam_tag_counts+true_bigam_tag_counts)
unknown_accuracy_bigram=unknown_correct_bigram/len(all_unknown_bigram)
print("Bigram Tagger Accuracy is {}%".format(bigram_accuracy))
print("Total unknowns is {}".format(len(all_unknown_bigram)))
print("Unknown Accuracy is {}%".format(unknown_accuracy_bigram))


print("------------CONFUSION MATRICES------------")

#A part of the below code has been re-used from my earlier assignment https://github.com/AKnightWing/bigram/blob/master/comp_ling.py
#Unigram Tagger Confusion Matrix
#Normalise both confusion dictionarues
for key in unigram_confusion:
    unigram_confusion[key]=100*unigram_confusion[key]/false_unigam_tag_counts

for key in bigram_confusion:
    bigram_confusion[key]=100*bigram_confusion[key]/false_bigam_tag_counts

firstrow=[' ']      #The first row in the 2D list
for key in tag_set:
    firstrow.append(key)

unigram_matrix=[]      #A n*n 2D list which stores only the skeleton of the matrix

for i in range(len(tag_set)+1):
    if i==0:
        unigram_matrix.append(firstrow)
    else:
        row=[]
        for j in range(len(tag_set)+1):
            if j==0:
                row.append(firstrow[i])
            else:
                try:
                    row.append(unigram_confusion[(tag_set[i]),(tag_set[j])])
                except Exception as e:
                    row.append("0")
        unigram_matrix.append(row)


bigram_matrix=[]

for i in range(len(tag_set)+1):
    if i==0:
        bigram_matrix.append(firstrow)
    else:
        row=[]
        for j in range(len(tag_set)+1):
            if j==0:
                row.append(firstrow[i])
            else:
                try:
                    row.append(bigram_confusion[(tag_set[i]),(tag_set[j])])
                except Exception as e:
                    row.append("0")
        bigram_matrix.append(row)

with open('unigram_confusion.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(unigram_matrix)
writeFile.close()
print("unigram_confusion.csv saved succesfully.")

with open('bigram_confusion.csv', 'w') as writeFile2:
    writer = csv.writer(writeFile2)
    writer.writerows(bigram_matrix)
writeFile2.close()
print("bigram_confusion.csv saved succesfully.")

t2=time.time()
print("Total time taken by program = {} seconds".format(t2-t1))