
 # Sentence Classifier

 ## Approach to feature selection
  1) consider a sentence, extract its parts of speech and dependencies
  2) for a given set of pos/dependency tags like ROOT which are unique and very important in a sentece, the GloVe embeddings are considered.  
  	 ex: if banana is the root in a sentence,   "banana" 's embeddings are considered

     Reason: Feeding sentences, as bag of words loses context, as padded/truncated sentences and feeding to a classier(maybe CNN) might need very large data set

    optinal: 
          a) since GloVe embeddings are 1x300 for any word, this can be redued by using tsne
                                            or
          b) use vecotor norm i.e L2 norm of the token/word 's vector which is a single value

 ## Classifier
SVM was used considering that a CNN would need more features than the ones currently in use which is 25.

 ## Global Variables
Global variables is not a very good idea. But in this case the NLP object is better off being global

`NLP` - spacy object , contains en_core_web_lg ~ 857MB

`EMBED_TAGS` - tags of pos/dependency whose embeddings from GloVe will be taken as input

`BOW_TAGS` - tags of pos/dependency using which a bag of words will be generated. 
 The bag of words counts the occurence of a type of pos tag. ex: the number of vers/adjectives etc

 ## Functions

`inc_bow(dictionary, key)` - increment count for making bag of words

`add_embeddings(tags_lst, tag_dict, debug_dict, token, tag, dim_reduce=False, dims=300)`  
For a given dependecy tag listed in **EMBED_TAGS**, load the GloVe embeddings value of its value.
as mentioned above,  if banana is the root in a sentence,   "banana" 's embeddings are considered

`process_sentence(sentence, dim_reduce=False, dims=300,echo=False)`
Create a bag of word dictionary for pos tags in BOW_TAGS, and dictionary of embeddings with add_embeddings

`process_data(x_raw, split_sentence=False, dim_reduce=False, dims=300)` 
Take a large dataset as input and process each sentence using aforementioned functions. Complex sentences may be split using spacy's built in functionalities.

`train(x_file, y_file, split_sentence=False, dim_reduce=False, dims=300)`
Train a given classifier using processed data.

`predict(sentences, classifier=None,split_sentence=False, dim_reduce=False, dims=300)` 
Make predictions using previously trained model

`load_pickle(filename="weights.pkl")` - load pickle from file

`save_pickle(data, filename="weights.pkl")` - save pickle from file