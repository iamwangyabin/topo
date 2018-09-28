import numpy as np

# Grid resolution
width = 40
height = 40

prob_factor = 100.0


def random_nodes(n):
    shape = (2 * height + 1, 2 * width + 1)
    n_max = shape[0] * shape[1]
    node_args = np.arange(1, n_max + 1)

    p_matrix = np.full(shape=shape, fill_value=prob_factor)
    p_matrix[1:-1, 1:-1] = 1.0
    #p_matrix[:10, :] = 1.0
    p_matrix /= p_matrix.sum()

    probs = p_matrix.T.ravel()
    nodes = np.random.choice(node_args, size=n,
                             replace=False, p=probs)
    return nodes


def random_nodes2(n):
    shape = (2 * height + 1, 2 * width + 1)
    n_max = shape[0] * shape[1]
    node_args = np.arange(1, n_max + 1)

    p_matrix = np.full(shape=shape, fill_value=prob_factor)
    p_matrix[1:-1, 1:-1] = 1.0
    #p_matrix[10:, :] = 1.0
    p_matrix /= p_matrix.sum()

    probs = p_matrix.T.ravel()
    nodes = np.random.choice(node_args, size=n,
                             replace=False, p=probs)
    return nodes


def random_config():
    n_fxtr_x = np.random.poisson(2)
    n_fxtr_y = np.random.poisson(3)
    n_load_y = np.random.poisson()
    config = {}
    config['FXTR_NODE_X'] = random_nodes(n_fxtr_x)
    config['FXTR_NODE_Y'] = random_nodes(n_fxtr_y)
    config['LOAD_NODE_Y'] = random_nodes2(n_load_y)
    config['LOAD_VALU_Y'] = [-1] * n_load_y  # Augmentation helps
    config['VOL_FRAC'] = np.random.normal(0.5, 0.1)

    return config
