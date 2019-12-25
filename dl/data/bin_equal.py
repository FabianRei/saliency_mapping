import numpy as np


def bin_equal(arr, num_bins, middle=1.11):
    sort_idxs = np.argsort(arr)
    unsort_idxs = np.argsort(sort_idxs)
    arr = arr[sort_idxs]
    if middle is not None:
        arrs = [arr[arr <= middle], arr[arr > middle]]
    else:
        arrs = [arr]
    result = []
    for i, a in enumerate(arrs):
        bins = np.round(num_bins/len(arrs))
        space = np.linspace(1/len(a), bins-1/len(a), num=len(a))
        space = space.astype(np.int)
        space += int(i*bins)
        result.extend(space)
    result = np.array(result)
    result = result[unsort_idxs]
    unsort_arr = arr[unsort_idxs]
    break_offs = [unsort_arr[result == i].max() for i in range(num_bins)]
    return result,  break_offs
