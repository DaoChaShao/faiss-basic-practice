from altair import Chart, X, Y, Color, Scale
from numpy import arange, array, linalg, dot
from pandas import DataFrame


def chart_points(query, index, top_n):
    """ Display a chart of similarity for the query vector and top_n nearest neighbors """

    # Use FAISS to find the top_n nearest neighbors for the query vector
    distances, indices = index.search(query, top_n)

    # Create a DataFrame with the results
    df = DataFrame({
        "Neighbor": arange(1, top_n + 1),
        "Distance": distances.flatten(),
        "Index": indices.flatten(),
        "Query": ["Query"] * top_n,
    })

    # 绘制散点图
    chart = Chart(df).mark_point(filled=True, size=100).encode(
        x="Neighbor:O",
        y="Distance:Q",
        color="Distance:Q",
        tooltip=["Index", "Distance", "Query"]
    ).properties(title="Query Vector and Top Nearest Neighbors")

    return chart
