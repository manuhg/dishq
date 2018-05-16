"""script to classify sentences """
#import numpy as np
import spacy
from sklearn.manifold import TSNE as tsne


DEP_TAGS = ['root', 'nsubj', 'det']
BOW_TAGS = ['verb', 'adv', 'adj', 'det', 'adv', 'aux', 'noun', 'pron', 'propn', 'nsubj', 'det']
print("Loading spacy en web large..")
NLP = spacy.load('en_core_web_lg')

def inc_bow(dictionary, key):
    """increment the value at key"""
    if dictionary.get(key):
        dictionary[key] += 1
#    else:
#        dictionary[key]=1


def process_sentence(sentence, dim_reduce=False, dims=300): #unicode encoded!
    """process each sentence i.e extract only required parts"""
    if  isinstance(sentence, 'unicode'):
        tokens = NLP(sentence)
        deps = {}
        bag_of_words = {key:0 for key in BOW_TAGS}
        for token in tokens:
            deptag = token.dep_.lower()
            postag = token.pos_.lower()
            inc_bow(bag_of_words, deptag)
            inc_bow(bag_of_words, postag)

            if (deptag in DEP_TAGS) and (deptag not in deps.keys()):
                #include only the first value of a kind

                embed_vec = token.lemma_.vector
                if dim_reduce and dims < embed_vec.size:
                    #if we want to reduce the glove vector to smaller dimension.
                    #default is 300
                    embed_vec = tsne(n_components=dims).fit_transform(embed_vec)

                    deps[deptag] = embed_vec
        return sum([sum(deps.values(), []), sum(bag_of_words.values(), [])], [])
    return list()

def process_data(x_raw, split_sentence=False, dim_reduce=False, dims=300):
    """extract required data for either training or for prediction """
    new_x = []
    for x in x_raw:
        if split_sentence:
            sentences=NLP(x)
            for sentence in sentences:
               new_x.append(process_sentence(sentence.string))
        else:
            new_x.append(process_sentence(sentence.string))

def train(x_file,y_file):
    try:
        x=open(x_file).read()
        y=open(y_file).read()
        x=process_data(x)
    except Exception as e:
        print("Unable to proceed with training.\n")
        print(e)
