import models as s
import sys
import pandas as pd
import numpy as np

from scipy.spatial import distance
from sentence_transformers import SentenceTransformer, util, InputExample, losses
from torch.utils.data import DataLoader


def compute(inputpath, lmodel, debug):
    """
    Name: compute
    Desc: select language model and similarity measurement then compute 
    """
    amendment_embeddings = []
    embeddings = []
    cos_predictions = []
    euc_predictions = []
    cos_final_predictions = []
    euc_final_predictions = []

    train_data, test_data = s.get_data(inputpath, debug)
    examples = s.data_to_input_examples(train_data)

    if lmodel == "bert":
        model = SentenceTransformer('all-mpnet-base-v2')

        s.finetune(examples, model)

        embeddings = model.encode(test_data["Input"].tolist())
        amendment_embeddings = model.encode([x for x in s.AMENDMENTS.values()])
    else:
        raise ValueError("Unknown language model")

    results = util.semantic_search(embeddings, amendment_embeddings)

    # cos_predictions = util.cos_sim(embeddings, amendment_embeddings)
    # euc_predictions = s.euclidean(embeddings, amendment_embeddings)
    
    # for sample in range(len(embeddings)):
    #     cos_final_predictions.append(np.argmax(cos_predictions[sample]))
    #     euc_final_predictions.append(np.argmax(euc_predictions[sample]))

    # s.evaluate(test_data, cos_final_predictions, "cosine")
    # s.evaluate(test_data, euc_final_predictions, "euclidean")
    print(results[:10])
    print("\nThe model has finished running.\n")
    # return results

def main():
    """
    Name: main
    Desc: ensure correct arugments have been passed in and then execute program
    """
    if len(sys.argv) != 4:
        raise Exception("usage: python models.py inputpath.csv [languageModel] [True/False]")

    input = sys.argv[1]
    lmodel = sys.argv[2]
    debug = sys.argv[3]

    compute(input, lmodel, debug)


if __name__ == '__main__':
    main()
