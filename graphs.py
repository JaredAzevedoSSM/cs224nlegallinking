"""
Name: graphs.py
Author(s): Jared Azevedo & Andres Suarez
Desc: 
"""
import numpy as np
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import sys


def embeddings_graph(embeddings: list, path: str): 
    """
    Name:
    Desc: 
    """
    matrix = np.zeros((len(embeddings), len(embeddings)))
    cosine_scores = util.cos_sim(embeddings, embeddings)
    
    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            matrix[i][j] = cosine_scores[i][j]
    
    plt.style.use('_mpl-gallery-nogrid')
    fig, ax = plt.subplots()
    ax.imshow(matrix)
    plt.savefig(f"{path}\str{embeddings}")


def main():
    """
    Name: main
    Desc: 
    """
    if len(sys.argv) != 2:
        raise Exception("usage: python organize_data.py inputpath [full/stripped]")

    inputpath = sys.argv[1]
    
    embeddings = None

    embeddings_graph(embeddings, inputpath)



if __name__ == "__main__":
    main()



