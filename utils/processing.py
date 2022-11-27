from .io import *
import numpy as np
import torchtext.vocab
from itertools import chain
from collections import Counter, OrderedDict


def preprocess_data(raw_data):
    for tag in raw_data:
        raw_data[tag] = process_dataset(raw_data[tag])
    return raw_data


def process_dataset(data):
    data['mask'] = [[True for w in s] for s in data['token']]
    return data


def process_input_dataset(data):
    data['mask'] = [[True for w in s] for s in data['token']]
    return data


#def process_dataset(data):
#    sentences, labels, postags, chars = data['token'], data['tag'], data['pos'], data['char']
#    p_sentences, p_labels, p_postags, p_chars, p_masks = [], [], [], [], []
#
#    for sentence, label, postag, char in zip(sentences, labels, postags, chars):
#
#        #p_sentence, p_label, p_postag, p_char, p_mask = ['<xxbos>'], [label[0]], ['<xxbos>'], [['<xxbos>']], [False]
#        p_sentence, p_label, p_postag, p_char, p_mask = [], [], [], [], []
#
#        for word, tag, pos, c in zip(sentence, label, postag, char):
#            """
#            if word[0].isupper():
#                p_sentence.append('<xxmaj>')
#                p_label.append(tag)
#                p_postag.append('<xxmaj>')
#                p_char.append(['<xxmaj>'])
#                p_mask.append(False)
#            """
#
#            p_sentence.append(word.lower())
#            p_label.append(tag)
#            p_postag.append(pos)
#            p_char.append([w.lower() for w in c])
#            p_mask.append(True)
#
#        """
#        p_sentence.append('<xxeos>')
#        p_label.append(p_label[-1])
#        p_postag.append('<xxeos>')
#        p_char.append(['<xxeos>'])
#        p_mask.append(False)
#        """
#
#        p_sentences.append(p_sentence)
#        p_labels.append(p_label)
#        p_postags.append(p_postag)
#        p_chars.append(p_char)
#        p_masks.append(p_mask)
#
#    return {'token':p_sentences, 'tag':p_labels, 'pos':p_postags, 'char':p_chars, 'mask':p_masks}


#def process_input_dataset(data):
#    sentences, postags, chars = data['token'], data['pos'], data['char']
#    p_sentences, p_postags, p_chars, p_masks = [], [], [], []
#
#    for sentence, postag, char in zip(sentences, postags, chars):
#
#        #p_sentence, p_postag, p_char, p_mask = ['<xxbos>'], ['<xxbos>'], [['<xxbos>']], [False]
#        p_sentence, p_label, p_postag, p_char, p_mask = [], [], [], [], []
#
#        for word, pos, c in zip(sentence, postag, char):
#            """
#            if word[0].isupper():
#                p_sentence.append('<xxmaj>')
#                p_postag.append('<xxmaj>')
#                p_char.append(['<xxmaj>'])
#                p_mask.append(False)
#            """
#
#            p_sentence.append(word.lower())
#            p_postag.append(pos)
#            p_char.append([w.lower() for w in c])
#            p_mask.append(True)
#
#        """
#        p_sentence.append('<xxeos>')
#        p_postag.append('<xxeos>')
#        p_char.append(['<xxeos>'])
#        p_mask.append(False)
#        """
#
#        p_sentences.append(p_sentence)
#        p_postags.append(p_postag)
#        p_chars.append(p_char)
#        p_masks.append(p_mask)
#
#    return {'token':p_sentences, 'pos':p_postags, 'char':p_chars, 'mask':p_masks}


def get_pretrained_embeddings(vocab, word2VecFile):
    word2Vec, word2VecVocab = readWord2Vector(word2VecFile)
    embeddings = []
    embed_dim = len(word2Vec[0])
    for word in vocab.get_itos():
        if word in word2VecVocab:
            embeddings.append(word2Vec[word2VecVocab[word]])
        else:
            embeddings.append(list(np.random.randn(embed_dim)))
    return torch.tensor(embeddings)


def get_pretrained_embedding_with_change_vocab(vocab, word2VecFile):
    word2Vec, word2VecVocab = readWord2Vector(word2VecFile)

    embeddings = []
    embed_dim = len(word2Vec[0])

    common_word, words = {}, {}
    for word in vocab.get_itos():
        if word in word2VecVocab:
            embeddings.append(word2Vec[word2VecVocab[word]])
            common_word.setdefault(word, len(common_word))
        else:
            words.setdefault(word, len(words))

    words.update({k:v+len(words) for k, v in common_word.items()})

    vocabulary = torchtext.vocab.vocab(words, min_freq=0)
    vocabulary.set_default_index( vocab.get_default_index() )

    return vocabulary, torch.tensor(embeddings)


def get_label_weights(raw_data, vocab):
    labels = []

    for name in raw_data:
        labels.extend(raw_data[name]['tag'])

    tag_pool = list(chain(*(labels)))
    tag_count = Counter(tag_pool)

    weights, count = [], []
    for tag in vocab.get_itos():
        weights.append(1/(tag_count[tag]))
        count.append(tag_count[tag])

    weights = np.array(weights)/np.sum(weights)
    return weights


def create_vocab(sentences, special_tkns=None, default_tkn=None, min_freq=1):
    tokens = Counter()
    for sentence in sentences:
        tokens.update(sentence)

    tokens_sorted_by_freq = sorted(tokens.items(), key=lambda x: x[1], reverse=True)
    tokens_dict = OrderedDict(tokens_sorted_by_freq)
    vocabulary = torchtext.vocab.vocab(tokens_dict, min_freq=min_freq, specials=special_tkns)
    if default_tkn:
        vocabulary.set_default_index(vocabulary[default_tkn])
    return vocabulary


def generate_vocab(raw_data, special_tkns, tags, default_tkn, min_freq=1):
    vocab = {}
    vocab['token'] = create_vocab(chain(*(raw_data[tag]['token'] for tag in tags)),
                                  special_tkns=special_tkns,
                                  default_tkn=default_tkn, min_freq=min_freq)
    vocab['pos'] = create_vocab(chain(*(raw_data[tag]['pos'] for tag in tags)),
                                special_tkns=special_tkns,
                                default_tkn=default_tkn)
    vocab['char'] = create_vocab( chain(*chain(*(raw_data[tag]['char'] for tag in tags))),
                                 special_tkns=special_tkns,
                                default_tkn=default_tkn)

    vocab['tag'] = create_vocab(chain(*(raw_data[tag]['tag'] for tag in tags)))
    return vocab


