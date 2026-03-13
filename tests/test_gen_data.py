import numpy as np

from bdselect.gen_data import gen_factor_data, gen_orthog_data
from bdselect.rng import make_rng

def main():

    rng = make_rng(1)
    dim_signal = 10

    X, y, beta = gen_factor_data(
        200,
        2,
        dim_signal,
        1000,
        1.5,
        1,
        rng
    )

    # X, y, beta = gen_orthog_data(
    #     200,
    #     dim_signal,
    #     1000,
    #     1,
    #     rng
    # )


    corr_matrix = np.corrcoef(X)
    idx = np.tril_indices(corr_matrix.shape[0], k=-1)
    print(np.max(corr_matrix[idx]))
    print(np.min(corr_matrix[idx]))

    print(np.mean(np.abs(corr_matrix[idx])))

    print(y.shape)
    print(beta.shape)

    # # Check IC
    # n, p = X.shape
    # C = X.T @ X / n
    # C11 = C[:dim_signal, :dim_signal]
    # C21 = C[dim_signal:, :dim_signal]
    # sign_beta = np.sign(beta)
    #
    # IC_matrix = np.abs( C21 @ np.linalg.inv(C11) @ sign_beta[:dim_signal].reshape(-1, 1) )
    #
    # counter = 0
    # for i in range(IC_matrix.shape[0]):
    #
    #     if i % 20 == 0:
    #         print(IC_matrix[i])
    #
    #     if IC_matrix[i] >= 1:
    #         counter += 1
    #         print(i)
    #
    # print(f"The number of IC violations is {counter}")
    #
    # print(C21[:20])



if __name__ == '__main__':
    main()
