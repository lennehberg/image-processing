import numpy as nm


def main():
    # create a random number generator instance
    rng = nm.random.default_rng()

    # generate a random array of size (9,1)
    rand_arr = rng.integers(10, size=(9,1))
    print(rand_arr)

    # reshape the array to a 3x3 matrix
    rand_mat = rand_arr.reshape(3, 3)
    print(rand_mat)
    print(rand_mat.shape)
    print(rand_mat.dtype)

    # multiply first column by third column
    res_vec = rand_mat[:, 0] * rand_mat[:, 2]
    print(res_vec)

    # create a new 9x9 matrix and fill it with res_vec
    mat_9x9 = nm.tile(res_vec, (9, 3))
    print(mat_9x9)

    # create a random 9x9 matrix
    rand_mat_9x9 = rng.integers(10, size=(9, 9))
    print(rand_mat_9x9)

    # stack matrices to create tensor
    tensor = nm.stack((mat_9x9, rand_mat_9x9), axis=0)
    print(tensor)
    print(nm.sum(tensor), nm.mean(tensor), nm.median(tensor))

    tensor[tensor > nm.mean(tensor)] = 0
    print(tensor)


if __name__ == "__main__":
    main()
