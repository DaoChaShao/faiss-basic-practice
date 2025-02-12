from numpy import random, ndarray

from utilities.tools import (faiss_index_creator,
                             faiss_index_adder,
                             faiss_index_search,
                             faiss_index_remover,
                             faiss_index_storager,
                             faiss_index_dropper,
                             faiss_index_loader)


def main():
    """ streamlit run main.py """
    dimensions: int = 256

    # create an empty index, which cannot be stored in parquet format
    index = faiss_index_creator(dimensions)

    # add some data to the index
    amount: int = 15
    faiss_index_adder(index, amount, dimensions)
    print(f"Index size: {index.ntotal}")

    # Give a random query vector
    amount: int = 1
    query: ndarray = random.rand(amount, dimensions).astype("float32")
    print(f"Query shape: {query.shape}")

    # Search for the top n nearest neighbors
    top_n: int = 3
    dis, idx = faiss_index_search(index, query, top_n)
    print(f"Top {top_n} distances(similarities): {dis}")
    print(f"Top {top_n} indices(position in the index): {idx}")

    # Delete vectors from the index
    ids: list[int] = [1, 2, 3]
    faiss_index_remover(index, ids)
    print(f"Index size after deletion: {index.ntotal}")

    # Storage the index
    file_name: str = "index"
    faiss_index_storager(index, file_name)

    # Drop the whole index
    faiss_index_dropper(index)
    print(f"Index size after dropping: {index.ntotal}")

    # Load the index from the storage
    index_new = faiss_index_loader(file_name)
    print(f"Index size after loading: {index_new.ntotal}")


if __name__ == "__main__":
    main()
