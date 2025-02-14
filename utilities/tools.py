from numpy import random, ndarray, array
import faiss


def faiss_index_creator(dimension: int, method: str = "Flat") -> faiss.Index:
    """ Creates a Faiss index based on the given dimension and method. """
    index = faiss.index_factory(dimension, method, faiss.METRIC_L2)
    # index = IndexFlatL2(dimension)
    return index


def faiss_index_adder(index: faiss, amount: int, dimension: int, vectors_type: str = "float32"):
    """ Adds vectors to a Faiss index. """
    vectors = random.rand(amount, dimension).astype(vectors_type)
    index.add(vectors)


def faiss_index_search(index: faiss, query: ndarray, top_n: int) -> tuple[ndarray, ndarray]:
    """ Searches a Faiss index for the k nearest neighbors of the given query vectors. """
    distances, indices = index.search(query, top_n)
    return distances, indices


def faiss_index_remover(index: faiss, ids: list[int]) -> None:
    """ Removes vectors from a Faiss index based on their ids. """
    index.remove_ids(array(ids))


def faiss_index_storager(index: faiss, file_name: str):
    """ Saves a Faiss index to a file. """
    faiss.write_index(index, f"{file_name}.faiss")


def faiss_index_dropper(index: faiss) -> None:
    """ Drops a Faiss index. """
    index.reset()


def faiss_index_loader(file_name: str) -> faiss.Index:
    """ Loads a Faiss index from a file. """
    return faiss.read_index(f"{file_name}.faiss")
