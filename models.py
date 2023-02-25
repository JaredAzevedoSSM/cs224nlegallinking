"""
Name: models.py
Author(s): Jared Azevedo & Andres Suarez
Desc: various language models and similarity measurements for matching legal texts with constitutional amendments
"""
import sys

from sentence_transformers import SentenceTransformer, util, InputExample, losses, evaluation
from torch.utils.data import DataLoader


def make_model(input, measurement):
    """
    Name: make_model
    Desc: if not using pretrained model, we want to create and train our own for the legal text similarity task
    Parameters:
        input - the training data we want to use; note that it may not be partitioned yet into train/test sets
        measurement - the measurement score we are using which may help us decide which loss to use
    """
    pass


def finetune(input, amendments, lmodel, measurement):
    """
    Name: finetune
    Desc: if using pretrained model, finetune for our legal text similarity task
    Parameters:
        input - the training data we want to use; note that it may not be partitioned yet into train/test sets
        lmodel - the model we are using
        measurement - the measurement score we are using which may help us decide which loss to use
    """
    train_dataloader = DataLoader(input, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(lmodel)

    # Need to rework this depending on how the input data looks like; evaluator is just to see progress during training
    evaluator = evaluation.EmbeddingSimilarityEvaluator(input, amendments, input) # last parameter should be input scores

    lmodel.fit(train_objectives=[(train_dataloader, train_loss)], evaluator=evaluator, epochs=1, evaluation_steps=1000, warmup_steps=100)



def compute(input, lmodel, measurement):
    """
    Name: compute
    Desc: select language model and similarity measurement then compute 
    """
    amendments = ["First Amendment", "Second Amendment", "Third Amendment", "Fourth Amendment", "Fifth Amendment", "Sixth Amendment",
                  "Seventh Amendment", "Eighth Amendment", "Ninth Amendment", "Tenth Amendment", "Eleventh Amendment", "Twelth Amendment",
                  "Thirteenth Amendment", "Fourteenth Amendment", "Fifteenth Amendment", "Sixteenth Amendment", "Seventeenth Amendment",
                  "Eighteenth Amendment", "Nineteenth Amendment", "Twentieth Amendment", "Twenty First Amendment", "Twenty Second Amendment",
                  "Twenty Third Amendment", "Twenty Fourth Amendment", "Twenty Fifth Amendment", "Twenty Sixth Amendment", 
                  "Twenty Seventh Amendment"]
    embeddings = None
    similarities = None

    if lmodel == "bert":
        model = SentenceTransformer('all-mpnet-base-v2')
        embeddings = model.encode(input)
    else:
        raise ValueError("Unknown language model")

    if measurement == "cosine":
        similarities = util.cos_sim(embeddings, amendments)
    else:
        raise ValueError("Unknown similarity measurement")
    
    all_combinations = []
    for i in range(len(similarities) - 1):
        for j in range(i + 1, len(similarities)):
            all_combinations.append([similarities[i][j], i, j])
    
    all_combinations = sorted(all_combinations, key=lambda x: x[0], reverse=True)

    print("Top-5 most similar pairs:")
    for _, i, j in all_combinations[:5]:
        print("{} \t {} \t {:.4f}".format(input[i], input[j], similarities[i][j]))


def main():
    """
    Name: main
    Desc: ensure correct arugments have been passed in and then execute program
    """
    if len(sys.argv) != 4:
        raise Exception("usage: python models.py [dataFile] [languageModel] [similarityMeasurement]")

    input = sys.argv[1]
    lmodel = sys.argv[2]
    measurement = str(sys.argv[3])

    compute(input, lmodel, measurement)


if __name__ == '__main__':
    main()
