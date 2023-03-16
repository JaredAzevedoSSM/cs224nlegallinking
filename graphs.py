"""
Name: graphs.py
Author(s): Jared Azevedo & Andres Suarez
Desc: 
"""
import numpy as np
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt

def embeddings_graph(): 
    pass



def main():
    """
    Name: main
    Desc: ensure correct arugments have been passed in and then execute program
    """
    if len(sys.argv) != 3:
        raise Exception("usage: python organize_data.py inputpath [full/stripped]")

    inputpath = sys.argv[1]
    cut = sys.argv[2]

    data = load(inputpath, cut)
    data = organize_and_label(data)
    data = separate(data)

    data = pd.DataFrame(data)

    data.to_csv(path_or_buf=f"../{cut}-compiled.csv", index=False)


if __name__ == "__main__":
    main()



