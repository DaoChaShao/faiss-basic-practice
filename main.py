from faiss import IndexFlatL2, IndexIVFFlat, METRIC_L2
from numpy import random, ndarray

from utilities.tools import (SeedNP,
                             faiss_index_creator,
                             faiss_index_adder,
                             faiss_index_search,
                             Timer, )


def main():
    """ streamlit run main.py """
    dimensions: int = 256
    amount_data: int = 1_000_000
    amount_query: int = 1
    top_n: int = 3
    nlist: int = 100
    probe: int = 50

    # create an empty index, which cannot be stored in parquet format
    index = faiss_index_creator(dimensions)

    with SeedNP(seed=9527):
        # add some data to the index
        faiss_index_adder(index, amount_data, dimensions)
        print(f"Index size: {index.ntotal}")

        # Give a random query vector
        query: ndarray = random.rand(amount_query, dimensions).astype("float32")
        print(f"Query shape: {query.shape}")

        # Search for the top n nearest neighbors
        with Timer(description="Brute-Force Search") as timer:
            dis, idx = faiss_index_search(index, query, top_n)
            # Print the results
            print(f"Top {top_n} distances(similarities): {dis}")
            print(f"Top {top_n} indices(position in the index): {idx}")
        print(timer)

        # Generate random vectors
        vector_ivf = random.rand(amount_data, dimensions).astype("float32")

        # Initialize the quantizer and the index
        quantizer = IndexFlatL2(dimensions)

        index_ivf = IndexIVFFlat(quantizer, dimensions, nlist, METRIC_L2)
        index_ivf.nprobe = probe

        # Train and add vectors to the index
        assert not index_ivf.is_trained
        index_ivf.train(vector_ivf)
        assert index_ivf.is_trained

        faiss_index_adder(index_ivf, amount_data, dimensions)

        # Perform search
        with Timer(description="IVF Search") as timer:
            d, i = faiss_index_search(index_ivf, query, top_n)
            # Print the results
            print(f"Top {top_n} distances(similarities): {d}")
            print(f"Top {top_n} indices(position in the index): {i}")
        print(timer)


if __name__ == "__main__":
    main()
