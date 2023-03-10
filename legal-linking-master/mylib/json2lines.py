from typing import Iterator, List, Dict
#from allennlp.data.fields import TextField, LabelField
#from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
#from allennlp.data.tokenizers import Token
#from allennlp.commands.train import *
#from allennlp.data.instance import Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
#from allennlp.data.dataset_readers.dataset_reader import DatasetReader
import json
import random
from tqdm import tqdm
import spacy
import re

from os.path import dirname


nlp = spacy.load('en')

# [
#     {
#         'text': "In 2008, the California Supreme Court held that...",
#         'meta': {
#             'doc_type': "opinion",
#             'id': 0
#             'source_url': "https://www.law.cornell.edu/supremecourt/text/12-144"
#         }
#         'matches': [["Fourteenth Amendment", "https://www.law.cornell.edu/constitution/amendmentxiv"]]
#     },
#     ...
# ]


def removeannoyingcharacters(text):
    """
    from https://stackoverflow.com/questions/6609895/efficiently-replace-bad-characters
    :param text: text that has annoying characters.
    :return: text with no annoying characters!
    """
    chars = {
        '’' : "'",
        "”": '"',
        '“': '"',
        '\x82': ',',        # High code comma
        '\x84': ',,',       # High code double comma
        '\x85': '...',      # Tripple dot
        '\x88': '^',        # High carat
        '\x91': "'",     # Forward single quote
        '\x92': "'",     # Reverse single quote
        '\x93': '"',     # Forward double quote
        '\x94': '"',     # Reverse double quote
        '\x95': ' ',
        '\x96': '-',        # High hyphen
        '\x97': '--',       # Double hyphen
        '\x99': ' ',
        '\xa0': ' ',
        '\xa6': '|',        # Split vertical bar
        '\xab': '<<',       # Double less than
        '\xbb': '>>',       # Double greater than
        '\xbc': '1/4',      # one quarter
        '\xbd': '1/2',      # one half
        '\xbe': '3/4',      # three quarters
        '\xca\xbf': '\x27',     # c-single quote
        '\xcc\xa8': '',         # modifier - under curve
        '\xcc\xb1': ''          # modifier - under line
    }

    def replace_chars(match):
        char = match.group(0)
        return chars[char]
    return re.sub('(' + '|'.join(chars.keys()) + ')', replace_chars, text)


class JsonConverter():

    def __init__(self) -> None:
        self._word_splitter = SpacyWordSplitter()

    def _read_const(self, file_dir):
        """
        This reads the constitution.json file and returns a dictionary.
        :param file_dir:
        :return:
        """

        # load constitutional data first.
        with open(file_dir + "/constitution.json") as f:
            infile = json.load(f)

        # there are some duplicate keys in the data.
        texts = set()

        # a dictionary that maps id to string.
        constitution = {}
        links = {}

        for k in infile.keys():
            txt = infile[k]["text"].strip()
            link = infile[k]["link"].strip()
            if len(txt) > 0:
                if txt not in texts:
                    # TODO: consider also passing metadata in
                    constitution[k] = " ".join(map(str, self._word_splitter.split_words(txt)))
                    links[k] = link
                    texts.add(txt)

        return constitution, links

    def _read(self, file_path: str, limit=-1):
        """
        file_path: must be in the same folder as constitution.json
        """
        counts = {"pos": 0, "neg": 0}

        with open(file_path) as f:
            lines = f.readlines()

        totalnum = min(len(lines), limit) if limit > -1 else len(lines)
            
        for i,line in tqdm(enumerate(lines), total=totalnum):

            if limit > -1 and i >= limit:
                break
            
            #print(line)
            grafs = json.loads(line)
            if not isinstance(grafs, list):
                grafs = [grafs]

            batch_text = [graf["text"] for graf in grafs]
            #batch_result = list(nlp.tokenizer.pipe(batch_text))            
            batch_result = self._word_splitter.batch_split_words(batch_text)

            for graf,toks in zip(grafs, batch_result):
                # print(graf)
                # doesn't make sense to have empty text?
                if len(toks) == 0:
                    graf["text"] = "empty"
                    continue
    
                # str. this is the slow part
                # t = " ".join(map(str, self._word_splitter.split_words(graf["text"])))
                t = " ".join(map(str, toks))
                graf_text = removeannoyingcharacters(t)

                # A set of all keys which are definitely negative
                # (according to the supervision we have)

                # if there are no matches, this won't run. NP
                if len(graf["matches"]) > 0:
                    grafkeys = set()
                    for match in graf["matches"]:
                        match_text, match_link, grafkey = match
                        grafkeys.add(grafkey)
                    counts["pos"] += 1
                    yield (graf_text, ",".join(grafkeys))
                else:
                    counts["neg"] += 1
                    yield(graf_text, "unmatched")

        print(counts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--infile', '-i', help='File of json to read in, probably called train/test/dev')
    parser.add_argument('--outfile', '-o', help='File to write to.')
    parser.add_argument('--limit', '-l', help='Limit on number of lines.', type=int, default=-1)
    parser.add_argument('--dumpconst', '-d', help='Dump the constitution as training data to this file.')

    args = parser.parse_args()

    ldr = JsonConverter()

    if args.dumpconst:

        print("writing constitution lines to", args.dumpconst)
        const, _ = ldr._read_const("data")
        with open(args.dumpconst, "w") as out:
            for k in const:
                seen = set()
                outline = "{}\t{}\n".format(const[k], k)
                out.write(outline)
                seen.add(outline)

                # split also by sentences.
                doc = nlp(const[k])
                for sent in doc.sents:
                    outline = "{}\t{}\n".format(sent, k)
                    if outline not in seen:
                        out.write(outline)
                        seen.add(outline)

    else:
        seen = set()
        with open(args.outfile, "w") as out:
            for trip in ldr._read(args.infile, limit=args.limit):
                outline = "\t".join(trip) + "\n"
                if outline not in seen:
                    out.write(outline)
                    seen.add(outline)
