import logging
import torchtext.data as data
import torchtext.datasets as datasets
from torchtext import vocab
import argparse
from autokeras import TextClassifier
import os.path
import shutil
import re

# Reference: https://github.com/Shawn1993/cnn-text-classification-pytorch
LOGGER = logging.getLogger("TextClassifier AutoKeras")
parser = argparse.ArgumentParser(description='CNN text classificer')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-path', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp"), help='number of embedding dimension [default: 128]')
args = parser.parse_args()


def sst(text_field, label_field, **kargs):
    train_data, dev_data, test_data = datasets.SST.splits(text_field, label_field, fine_grained=True)
    print(type(train_data))
    print(len(train_data))

    print(type(train_data[0]))
    ex = train_data[0]
    print(train_data.fields.items())
    print(ex.text)
    print(ex.label)

    vec = vocab.Vectors('glove.6B.100d.txt', 'glove_embedding/')
    text_field.build_vocab(train_data, dev_data, test_data, max_size=100000, vectors=vec)
    label_field.build_vocab(train_data, dev_data, test_data)
    print(text_field.vocab.vectors[text_field.vocab.stoi['the']])
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
                                        (train_data, dev_data, test_data),
                                        batch_sizes=(args.batch_size,
                                                     len(dev_data),
                                                     len(test_data)),
                                        **kargs)
    return train_iter, dev_iter, test_iter


def clean_path(path):
    try:
        shutil.rmtree(path)
        print("directory cleaned")
    except OSError as e:
        pass
    print("creating the directory %s" % path)
    os.makedirs(path)


def clean(text):
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text) # remove non alphanumeric character
    text = re.sub(r'https?:/\/\S+', ' ', text) # remove links
    return text.strip()


LOGGER.debug("Loading data...")
print("Loading data...")

text_field = data.Field(lower=True, include_lengths=True, use_vocab=True)
label_field = data.Field(sequential=False)
train_iter, dev_iter, test_iter = sst(text_field, label_field, device=-1, repeat=False)

for batch in train_iter:
    feature, target = batch.text, batch.label
    print(feature[0])
    print(feature[0].shape)
    print(feature.shape)
    print(target)
    print(target.shape)
    break
args.embed_num = len(text_field.vocab)
args.class_num = len(label_field.vocab) - 1
print(text_field.vocab.vectors.shape)


path = args.path
clean_path(path)
clf = TextClassifier(verbose=True, path=path)
clf.fit(train_iter, dev_iter, test_iter, args.embed_num, args.class_num, time_limit=12 * 60 * 60)






