"""script to classify sentences """
#import numpy as np
import pickle
import spacy
from sklearn import svm
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


def process_sentence(sentence, dim_reduce=False, dims=300):
    """process each sentence i.e extract only required parts
       input sentence must be enoced in utf-8 """
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
    for row in x_raw:
        if split_sentence:
            sentences = NLP(row)
            for sentence in sentences:
                new_x.append(process_sentence(sentence.string, dim_reduce, dims))
        else:
            new_x.append(process_sentence(sentence.string, dim_reduce, dims))

def load_pickle(filename="weights.pkl"):
    """load saved, trained weights from file"""
    with open(filename, 'rb') as pfile:
        model = pickle.load(pfile)
        return model
    return None

def save_pickle(data, filename="weights.pkl"):
    """save  trained weights to file"""
    with open(filename, 'wb') as pfile:
        pickle.dump(data, pfile)
        pfile.close()

def train(x_file, y_file, split_sentence=False, dim_reduce=False, dims=300):
    """train the model"""
    try:
        x = open(x_file).read()
        y = open(y_file).read()
        x = process_data(x, split_sentence, dim_reduce, dims)
        classifier = svm.LinearSVC()
        classifier.fit(x,y)
        save_pickle(classifier)
        return classifier
    except Exception as e:
        print("Unable to proceed with training.\n")
        print(e)
    return None

def predict(sentence, classifier=None):
    """sentence must be a unicode encoded string"""
    if classifier is None:
        classifier=load_pickle()
    classifier.decision_function(process_data(sentence))
