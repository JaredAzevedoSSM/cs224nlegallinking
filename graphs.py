"""
Name: graphs.py
Author(s): Jared Azevedo & Andres Suarez
Desc: 
"""
import numpy as np
from sentence_transformers import util
import matplotlib.pyplot as plt
import sys
import re


def read_embeddings(path: str):
    """
    Name: read_embeddings
    Desc: 
    """
    embeddings = []

    with open(path, 'r') as f:
        lines = f.read().splitlines()
        num_newlines = 0
        embedding = []

        for line in lines:
            values = line.split()

            for value in values:
                value = re.sub("[\[\]]", "", value)
                embedding.append(float(value))

            if len(values) == 0:
                num_newlines += 1
            
            if num_newlines == 2:
                embeddings.append(embedding)
                embedding = []
                num_newlines = 0

    return embeddings


def embeddings_graph(inputpath: str, outputpath: str): 
    """
    Name: embeddings_graph
    Desc: 
    """
    embeddings = read_embeddings(inputpath)

    matrix = np.zeros((len(embeddings), len(embeddings)))
    cosine_scores = util.cos_sim(embeddings, embeddings)
    
    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            matrix[i][j] = cosine_scores[i][j]
    
    plt.style.use('_mpl-gallery-nogrid')
    fig, ax = plt.subplots()
    ax.imshow(matrix)
    plt.savefig(f"{outputpath}.jpg")


def main():
    """
    Name: main
    Desc: 
    """
    if len(sys.argv) != 3:
        raise Exception("usage: python graphs.py inputpath outputpath")

    inputpath = sys.argv[1]

    outputpath = sys.argv[2]

    embeddings_graph(inputpath, outputpath)


if __name__ == "__main__":
    main()
