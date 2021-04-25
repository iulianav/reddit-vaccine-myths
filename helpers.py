import matplotlib.pyplot as plt
import pandas as pd
import re
import seaborn as sns
import spacy

from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from pycontractions import Contractions
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


def get_frequencies_df(series, label, min_freq=1):
    """Computes a DataFrame from Series values and their occurence frequencies.
    
    Args:
        series: A given pandas Series.
        label: The name of the variable contained in the series.
        min_freq: Only values with a frequency higher than min_freq
        will be returned.

    Returns:
        A pandas DataFrame containing 2 columns: the value of
        the given variable and its occurence counts. For example:

        {'variable_name': ['dog', 'cat', 'horse', 'bear']],
         'variable_name_count': [8, 8, 4, 2]}

    """
    
    # Get the values.
    keys = list(series.value_counts().keys())

    # Get the frequency counts.
    values = series.value_counts().values

    freq_df =  pd.DataFrame({label: keys, '{}_count'.format(label): values})
    
    # Keep only values that occur at least min_freq times in the Series.
    freq_df = freq_df[freq_df['{}_count'.format(label)] >= min_freq]
    
    return freq_df
    
    
def get_ngrams_freq(texts, ngram_range=(1, 1), min_freq=5):
    """Computes a dictionary of n-grams and their occurence frequencies. 
    
    Args:
        texts: A list of text documents.
        ngram_range: The range of n-grams to include.
        min_freq: Only n-grams with a frequency higher than min_freq
        will be returned.

    Returns:
        A dict mapping n-grams to the corresponding occurence count.
        For example:

        {'immune system': 100,
         'vaccine measles': 50}

    """
    
    # Convert collection of text documents to a matrix of bigram counts.
    min_gram, max_gram = ngram_range
    cv = CountVectorizer(ngram_range=(min_gram, max_gram))

    ngrams = cv.fit_transform(texts)

    # Get n-grams vocabulary.
    ngram_vocab = cv.vocabulary_

    count_values = ngrams.toarray().sum(axis=0)

    ngrams_freq = {}
    # Output n-grams with a frquency higher than 5.

    # List of ngrams and counts sorted in descending order of counts.
    ngrams_counts_list = sorted(
        [(count_values[i], k) for k,i in ngram_vocab.items()],
        reverse=True,
        )

    for ng_count, ng_text in ngrams_counts_list:
        if ng_count > min_freq:
            ngrams_freq[ng_text] = ng_count
            
    return ngrams_freq


def get_tokens_series(txt_series,
                      contractions=True,
                      lemmatize=True,
                      stop_words=None,
                      token_min_len=5):
    """Computes a pandas Series containing lists of tokens. 
    
    Args:
        txt_series: A series of text documents to be tokenized.
        contractions: Optional; If set to True, it expands contractions
        e.g. don't -> do not.
        lemmatize: Optional; If set to True, the tokens are lemmatized.
        stop_words: Optional; If not None, the passed stop_words list
        will be used to remove stop words.
        token_min_len: Only keep tokens that are at least token_min_len
        characters long.

    Returns:
        A pandas Series containing each tokenized text document as a list
        of tokens. Each row is a list of tokens. For example:

        0          [health, canada, approves, astrazeneca, covid]
        1          [covid, canada, passport, certainty, ethicist]
        2            [coronavirus, variant, could, canada, third]

    """
    
    if contractions:
        # Initialize word contractions expander.
        cont = Contractions(api_key="glove-twitter-100")

        # Expand contractions e.g. `don't` -> `do not`.
        expanded_text = pd.Series(list(cont.expand_texts(txt_series.tolist())))
        txt_series = expanded_text

    # Remove emails.
    txt_series = txt_series.apply(lambda x: re.sub(r'\S*@\S*\s?', ' ', x))

    # Remove subreddit mentiones.
    txt_series = txt_series.apply(lambda x: re.sub(r'r/\w+',' ', x))

    # Remove user mentiones.
    txt_series = txt_series.apply(lambda x: re.sub(r'/u/\w+',' ', x))

    # Remove hashtags.
    txt_series = txt_series.apply(lambda x: re.sub(r'#(\d|\w)+',' ', x))

    # Remove links.
    txt_series = txt_series.apply(lambda x: re.sub(r'http\S+',' ', x))

    # Remove digits and words containing digits.
    txt_series = txt_series.apply(lambda x: re.sub('\w*\d\w*',' ', x))

    if not stop_words:
        # English stop words list.                                            
        stop_words = stopwords.words('english')

    # Preprocess (punctuation removal, digits removals, etc)
    # and tokenize the text.
    tokens_series = txt_series.apply(
        lambda x: simple_preprocess(x, deacc=True)
        )

    if lemmatize:
        # Word lemmatizer.
        lemmatizer = WordNetLemmatizer()
        # Lemmatize the tokens.
        tokens_series = tokens_series.apply(
            lambda x: [lemmatizer.lemmatize(item) for item in x]
            )

    # Remove stop words.
    tokens_series = tokens_series.apply(
        lambda x: [item for item in x if item not in stop_words]
        )

    # Keep only terms longer than 5 characters.
    tokens_series = tokens_series.apply(
        lambda x: [item for item in x if len(item) >= token_min_len]
        )
    
    return tokens_series
    

def get_sentiment(text, analyzer):
    """Compute the sentiment label of a document
    based on a given VADER analyzer.
    
    Args:
        text: A given text document.
        analyzer: VADER analyser for sentiment analysy.

    Returns:
        A string representing the identified sentiment
        based on the compound score returned by the analyser.
        Possible values are: 'Positive', 'Negative' or
        'Neutral'.

    """

    compound_score = analyzer.polarity_scores(text)['compound']
    
    if compound_score  >= 0.05:
        return 'positive'
    
    if compound_score <= -0.05:
        return 'negative'
    
    return 'neutral'


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Lemmatize a given list of text documents.
    
    Retrieves a list of tokens with the allowed POS tags. 
    
    Args:
        texts: A list of text documents.
        allowed_postags: The allowed token POS tags.

    Returns:
        A list of lists. Each list contains the lemmatized tokens of the
        allowed POS tags for one document. For example:

        [['regard', 'childhood', 'schedule', 'reason'],
         ['check', 'paraphrase', 'multi', 'people'],
         ['childhood', 'statement'],
         ['mercury', 'disagreeing', 'tracking', 'conversation']]

    """
    
    # Initialize spacy 'en' model, keeping only tagger component.
    nlp = spacy.load('en', disable=['parser', 'ner'])

    texts_out = []
    for sent in texts:
        doc = nlp(' '.join(sent)) 
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags]
            )
    return texts_out


def make_bigrams(texts, bigram_mod):
    """Return the bi-gram models for a list of text documents.
    
    Args:
        texts: A list of text documents.
        bigram_mod: A given bi-gram model.

    Returns:
        A list of lists. Each list contains the bi-gram model tokens for
        a document. For example:

        [['health', 'canada', 'approves', 'astrazeneca', 'covid'],
         ['covid', 'canada', 'passport', 'certainty', 'ethicist'],
         ['coronavirus', 'variant', 'could', 'canada', 'third']]
         
    """
    
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts, bigram_mod, trigram_mod):
    """Return the tri-gram models for a list of text documents.
    
    Args:
        texts: A list of text documents.
        bigram_mod: A given bi-gram model.
        trigram_mod: A given tri-gram model.

    Returns:
        A list of lists. Each list contains the tri-gram model tokens for
        a document. For example:

        [['health', 'canada', 'approves', 'astrazeneca', 'covid'],
         ['covid', 'canada', 'passport', 'certainty', 'ethicist'],
         ['coronavirus', 'variant', 'could', 'canada', 'third']]

    """
    
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def plot_top_terms_heatmap(texts, series, terms_num=50):
    """Plots a heatmap of the top tokens in a Series of tokens.
    
    Args:
        texts: A list of text documents based on which co-occurence matrix
        is computed.
        series: A series containing the documents' tokens.
        terms_num: Top terms_num to consider based on occurence counts.

    """
    
    # Convert collection of text documents to a matrix of token counts.
    cv = CountVectorizer(ngram_range=(1,1), stop_words='english')

    # Matrix of token counts.
    X = cv.fit_transform(texts)

    # Matrix manipulation for terms co-occurence counts.
    Xc = (X.T * X) 

    # Set the diagonals to be zeroes as it's pointless to be 1.
    Xc.setdiag(0)

    # Get the unique terms names as features.
    names = cv.get_feature_names()

    # Set the columns and index to the terms names
    # of the co-occurence countsmatrix.
    co_occurence_df = pd.DataFrame(data=Xc.toarray(),
                                   columns=names,
                                   index=names)

    # Get the most common terms.
    top_terms = series.value_counts()[:terms_num].keys()

    # Keep only the most common terms saved by the CountVectorizer model.
    top_terms = top_terms[top_terms.isin(co_occurence_df.columns)]

    # Keep only the counts for the most frequent terms.
    heatmap_df = co_occurence_df.loc[top_terms, top_terms]
    
    # Plot co-occurence heatmap for top terms.
    plt.figure(figsize=(20, 10))
    plt.title('Frequent terms co-occurence heatmap', size=20)
    sns.heatmap(heatmap_df,
                cmap='YlGnBu',
                robust=True,
                square=True,
                cbar_kws={'label': 'Co-occurence count'})
    

def plot_topics_wordclouds(lda_model, num_topics):
    """Plots a wordcloud for each topic in a given LDA model.
    
    Args:
        lda_mode: A given LDA model.
        num_topics: The number of topics in modelled by the LDA model.

    """
    
    topic = 0 # Initialize counter
    while topic < num_topics:
        # Get topics and frequencies and store in a dictionary structure.
        topic_words_freq = dict(lda_model.show_topic(topic, topn=50))

        # Generate Word Cloud for topic using frequencies.
        wordcloud = WordCloud().generate_from_frequencies(topic_words_freq)
        topic += 1

        # Plot wordclowd.
        plt.figure(figsize=(15,7))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title('Topic {}'.format(str(topic)), size=25)
        plt.axis('off')
        plt.show()


def remove_stopwords(texts, stop_words):
    """Tokenize and remove stop words from a given list of text documents.
    
    Retrieves a list of tokens with the allowed POS tags. 
    
    Args:
        texts: A list of text documents.
        stop_words: Stop words to remove.

    Returns:
        A list of lists. Each list contains the tokens of the for one document
        excluding stop words. For example:

        [['regard', 'childhood', 'schedule', 'reason'],
         ['check', 'paraphrase', 'multi', 'people'],
         ['childhood', 'statement'],
         ['mercury', 'disagreeing', 'tracking', 'conversation']]

    """
    tokenized_documents = []
    for doc in texts:
        current_word_list = []
        for word in simple_preprocess(str(doc)):
            if word not in stop_words:
                current_word_list.append(word)
        tokenized_documents.append(current_word_list)

    return tokenized_documents


def run_lda_models(corpus,
                   dictionary,
                   texts,
                   min_topics_num=2,
                   max_topics_num=40):
    """Run an LDA model for each number of topics in the specified ranfe.
    
    Retrieves a list of tuples containing the number of topics and equivalent
    coherence score. 
    
    Args:
        corpus: Bag of Words representation of documents.
        dictionary: Dictionary of terms.
        texts: Text documents represented as lists of tokens.
        min_topics_num: Minimum number of topics to consider.
        max_topics_num: Maximum number of topics to consider.

    Returns:
        A list of tuples. Each tuple contains the number of topics and corresponding
        coherence score. For example:

        [(5, 0.40)),
         (6, 0.41)),
         (7, 0.35),
         (8, 0.34))]

    """
    
    coherence_scores = list()
    for num_topics in range(min_topics_num, max_topics_num):
        # Build LDA model.
        lda_model = LdaModel(corpus=corpus,
                             id2word=dictionary,
                             num_topics=num_topics, 
                             random_state=100,
                             update_every=1,
                             chunksize=100,
                             passes=10,
                             alpha='auto',
                             eta='auto',
                             iterations=400,
                             per_word_topics=True)

        # Compute and store coherence score.
        coherence_model_lda = CoherenceModel(model=lda_model,
                                             texts=texts,
                                             dictionary=dictionary,
                                             coherence='c_v')

        coherence_lda = coherence_model_lda.get_coherence()
        coherence_scores.append((num_topics, coherence_lda))

    return coherence_scores


def sent_to_words(texts):
    """Tokenize a list of documents and return a generator
    yielding the lists of tokens.
    
    The tokenization process includes removing punctuation,
    end of line characters, digits, etc.
        
    Args:
        texts: A list of text documents.

    Returns:
        A generatore. Each item in the generator is a list of
        preprocessed tokens. For example:

        [['regard', 'childhood', 'schedule', 'reason'],
         ['check', 'paraphrase', 'multi', 'people'],
         ['childhood', 'statement'],
         ['mercury', 'disagreeing', 'tracking', 'conversation']]

    """

    for sent in texts:
        # deacc=True removes punctuations.
        yield(simple_preprocess(str(sent), deacc=True))
