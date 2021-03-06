"""script to classify sentences """
#import numpy as np
import sys
import pickle
import spacy
from sklearn import svm
from sklearn.manifold import TSNE as tsne
from sys import stdin

EMBED_TAGS = ['root', 'nsubj','adv', 'adj', 'det']
#tags whose values' embeddings will be considered
BOW_TAGS = ['verb', 'adv', 'adj', 'det', 'aux', 'noun', 'pron', 'propn', 'intj', 'nsubj', 'det']
#tags to make bag of words


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

def inc_bow(dictionary, key):
    """increment the value at key"""
    if key.strip() in dictionary.keys():
        dictionary[key] += 1
#    else:
#        dictionary[key]=1

def add_embeddings(tags_lst, tag_dict, debug_dict, token, tag, dim_reduce=False, dims=300):
    """given a tag(pos or dependency), add its embeddings value to dictionary"""
    if (tag in tags_lst) and (tag not in tag_dict.keys()):
            #include only the first value of a kind
            embed_vec = None
            if dims <= 1:
                dims=1
                embed_vec = [token.vector_norm] # l2 norm
            elif dim_reduce and dims < 300:
                #if we want to reduce the glove vector to smaller dimension. default is 300
                #dims = 3 if dims>4 else dims #because of a weird bug with tsne
                embed_vec = list(tsne(n_components=dims).fit_transform([token.vector,token.vector])[0])
            else:
                embed_vec = list(token.vector)
            tag_dict[tag] = embed_vec
            debug_dict[tag]=token.text

def process_sentence(sentence, dim_reduce=False, dims=300, echo=False):
    """process each sentence i.e extract only required parts i.e pos and dependency"""
    tokens = NLP(sentence)
    embeddings = {}
    embeds_for_debug = {}
    bag_of_words = {key:0 for key in BOW_TAGS}
    for token in tokens:
        deptag = token.dep_.lower()
        postag = token.pos_.lower()
        if echo:
            print(token.text, deptag, postag)

        inc_bow(bag_of_words, deptag)
        inc_bow(bag_of_words, postag)

        add_embeddings(EMBED_TAGS, embeddings, embeds_for_debug, token, deptag, dim_reduce, dims)
        add_embeddings(EMBED_TAGS, embeddings, embeds_for_debug, token, postag, dim_reduce, dims)


    for d in EMBED_TAGS:
        if d not in embeddings.keys():
            embeddings[d] = [0]*dims
            embeds_for_debug[d] = ''

    if echo:
        print("\nDeps:\n", embeds_for_debug, "\nbow:\n", bag_of_words)
    return sum([sum(list(embeddings.values()), []), list(bag_of_words.values())], [])


def process_data(x_raw, split_sentence=False, dim_reduce=False, dims=300):
    """extract required data for either training or for prediction """
    new_x = []
    for row in x_raw:
        if split_sentence:
            sentences = NLP(row)
            for sentence in sentences.sents:
                new_x.append(process_sentence(sentence.string, dim_reduce, dims))
        else:
            new_x.append(process_sentence(row, dim_reduce, dims))
    return new_x
def train(x_file, y_file, classifier, split_sentence=False, dim_reduce=False, dims=300):
    """train the model"""
    try:
        x = open(x_file).read().split("\n")
        y = open(y_file).read().split("\n")
        x = process_data(x, split_sentence, dim_reduce, dims)
        #classifier = svm.LinearSVC()
        classifier.fit(x, y)
        save_pickle(classifier)
        return classifier
    except Exception as e:
        print("Unable to proceed with training.\n")
        print(e)
    return None

def predict(sentences, classifier=None, split_sentence=False, dim_reduce=False, dims=300):
    """sentence must be a unicode encoded string"""
    if classifier is None:
        classifier = load_pickle()
    return classifier.decision_function(process_data(sentences, split_sentence, dim_reduce, dims))
def main():
    """main function"""
    classes = ['AIMX','BASE','CONT','MISC','OWNX']
    classifier = load_pickle()
    if classifier:
        print("Sucessfully loaded classifier")
    else:
        print("Failed to load classifier")
        sys.exit(0)
    sentences = []
    print("Enter one or more sentences, hit enter for new sentence, Ctrl-D when you are done.")
    for line in stdin:
        sentences.append(line)
    if sentences :
        predictions=predict(sentences, classifier, False, True, 3)
        for p in predictions:
            print(p,classes.index(max(p)))

print("Loading spacy en web large..")
NLP = spacy.load('en_core_web_lg')
if __name__ == "__main__":
    main()
