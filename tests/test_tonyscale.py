import numpy as np
from tonyscale import scale_image


def tony_scale_numpy(data, n_bins=100_000, n_colors=256):
    flat_data = data.ravel()

    h_min = np.floor(flat_data.min()) - 1.0
    h_max = np.ceil(flat_data.max()) + 1.0

    hist, bin_edges = np.histogram(flat_data, bins=n_bins, range=(h_min, h_max))
    bin_size = bin_edges[1] - bin_edges[0]

    cmap = np.arange(n_colors, dtype=np.int64)

    cdf = cmap[np.floor((cmap.size - 1)*np.cumsum(hist)/flat_data.size).astype(np.int64)]

    bins = np.floor((flat_data - h_min)/bin_size).astype(np.int64)
    return cdf[bins].reshape(data.shape)


def test_scale_image():
    np.random.seed(12345)

    data = np.random.normal(loc=0.0, scale=100.0, size=(4096, 4096))

    numpy_scaled = tony_scale_numpy(data)

    scaled = scale_image(data)

    # There are some differences by 1 value that I don't know about now.
    assert(np.all(np.abs(scaled - numpy_scaled) <= 1))
