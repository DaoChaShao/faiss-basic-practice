from faiss import Index, index_factory, METRIC_L2, write_index, read_index
from numpy import random, ndarray, array
from time import perf_counter


def faiss_index_creator(dimension: int, method: str = "Flat") -> Index:
    """ Creates a Faiss index based on the given dimension and method. """
    index = index_factory(dimension, method, METRIC_L2)
    # index = IndexFlatL2(dimension)
    return index


def faiss_index_adder(index, amount: int, dimension: int, vectors_type: str = "float32"):
    """ Adds vectors to a Faiss index. """
    vectors = random.rand(amount, dimension).astype(vectors_type)
    index.add(vectors)


def faiss_index_search(index, query: ndarray, top_n: int) -> tuple[ndarray, ndarray]:
    """ Searches a Faiss index for the k nearest neighbors of the given query vectors. """
    distances, indices = index.search(query, top_n)
    return distances, indices


def faiss_index_remover(index, ids: list[int]) -> None:
    """ Removes vectors from a Faiss index based on their ids. """
    index.remove_ids(array(ids))


def faiss_index_storager(index, file_name: str):
    """ Saves a Faiss index to a file. """
    write_index(index, f"{file_name}.faiss")


def faiss_index_dropper(index) -> None:
    """ Drops a Faiss index. """
    index.reset()


def faiss_index_loader(file_name: str) -> Index:
    """ Loads a Faiss index from a file. """
    return read_index(f"{file_name}.faiss")


class SeedNP(object):
    """ A class to seed the numpy random number controller. """

    def __init__(self, seed: int):
        self._seed = seed

    def __enter__(self):
        self._state = random.get_state()
        random.seed(self._seed)
        print(f"Seeding numpy with {self._seed}.")

    def __exit__(self, exc_type, exc_val, exc_tb):
        random.set_state(self._state)


class Timer(object):
    """ A class to measure the time of a code block. """

    def __init__(self, precision: int = 5, description: str = None):
        self._precision = precision
        self._desc = description
        self._elapsed_time = None

    def __enter__(self):
        self._start = perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end = perf_counter()
        self._elapsed_time = self._end - self._start

    def __repr__(self):
        if self._elapsed_time is None:
            return "Timer not started yet."
        else:
            return f"{self._desc} elapsed: {self._elapsed_time:.{self._precision}f} seconds"
