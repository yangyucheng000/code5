from scipy.special import expit as sigmoid
import igraph as ig
import numpy as np
import random

def generate_graph(d, e, graph_type, seed=None):
    '''
    :param d: num of nodes
    :param e: num of edges
    :param graph_type: ER, SF
    :param seed: random seed
    :return: binary adjacency matrix of DAG
    '''
    def set_random_seed(seed):
        random.seed(seed)
        np.random.seed(seed)

    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    def Erdos_Renyi(d, e):
        G_und = ig.Graph.Erdos_Renyi(n=d, m=e)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
        return B

    def Scale_Free(d, e):
        G_und = ig.Graph.Barabasi(n=d, m=int(round(e/d)))
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
        return B

    set_random_seed(seed)
    if graph_type == 'ER':
        B = Erdos_Renyi(d, e)
    elif graph_type == 'SF':
        B = Scale_Free(d, e)
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


def simulate_data(B, n, sem_type, noise_scale=None):
    '''
    :param B: binary adjacency matrix of DAG
    :param n: num of samples
    :param sem_type: linear: gauss, exp, gumbel, uniform, logistic, poisson; nonlinear: mlp, mim, gp, gp-add
    :param noise_scale: scale parameter of additive noise, default all ones
    :return: [n, d] sample matrix
    '''
    def is_dag(W):
        G = ig.Graph.Weighted_Adjacency(W.tolist())
        return G.is_dag()

    def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
        W = np.zeros(B.shape)
        S = np.random.randint(len(w_ranges), size=B.shape)  # which range
        for i, (low, high) in enumerate(w_ranges):
            U = np.random.uniform(low=low, high=high, size=B.shape)
            W += B * (S == i) * U
        return W

    def simulate_nonlinear_sem(B, n, sem_type, noise_scale=None):
        def _simulate_single_equation(X, scale):
            z = np.random.normal(scale=scale, size=n)
            pa_size = X.shape[1]
            if pa_size == 0:
                return z
            elif sem_type == 'gp':
                from sklearn.gaussian_process import GaussianProcessRegressor
                gp = GaussianProcessRegressor()
                x = gp.sample_y(X, random_state=None).flatten() + z
            elif sem_type == 'gp-add':
                from sklearn.gaussian_process import GaussianProcessRegressor
                gp = GaussianProcessRegressor()
                x = sum([gp.sample_y(X[:, i, None], random_state=None).flatten()
                         for i in range(X.shape[1])]) + z
            else:
                raise ValueError('unknown sem type')
            return x

        d = B.shape[0]
        if noise_scale is None:
            scale_vec = np.ones(d)
        elif np.isscalar(noise_scale):
            scale_vec = noise_scale * np.ones(d)
        else:
            if len(noise_scale) != d:
                raise ValueError('noise scale must be a scalar or has length d')
            scale_vec = noise_scale
        X = np.zeros([n, d])
        G = ig.Graph.Adjacency(B.tolist())
        ordered_vertices = G.topological_sorting()
        assert len(ordered_vertices) == d
        for j in ordered_vertices:
            parents = G.neighbors(j, mode=ig.IN)
            X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j])
        return X

    X = simulate_nonlinear_sem(B, n, sem_type, noise_scale)
    return X


def get_Mg_graph(B, ma_num, mi_num):
    Ma_B = B.copy()
    Mi_B = B.copy()
    potential_child = np.where(np.sum(Ma_B, axis=0) == 0)[0]  # root
    if potential_child.shape[0] > ma_num:
        children = random.sample(list(potential_child), ma_num)
    else:
        children = potential_child

    for child in children:
        # Ma_B
        row = np.zeros((1,Ma_B.shape[1]))
        row[0,child] = 1
        colum = np.zeros((Ma_B.shape[0]+1,1))
        Ma_B = np.append(Ma_B, row, axis=0)
        Ma_B = np.append(Ma_B, colum, axis=1)
        # Mi_B
        rows = np.zeros((mi_num,Mi_B.shape[1]))
        rows[:,child] = 1
        Mi_B = np.append(Mi_B, rows, axis=0)
        columns = np.zeros((Mi_B.shape[0], mi_num))
        Mi_B = np.append(Mi_B, columns, axis=1)

    return Ma_B, Mi_B


def get_Mg_data(v, X, d, mi_num=3):
    X = X.copy()
    for m in range(v-1,d-1,-1):
        ma_data = X[:,m].reshape((-1,1))
        hidden = 100
        W1 = np.random.uniform(low=0.5, high=2.0, size=[1, hidden])
        W1[np.random.rand(*W1.shape) < 0.5] *= -1
        W2 = np.random.uniform(low=0.5, high=2.0, size=[hidden, mi_num])
        W2[np.random.rand(*W2.shape) < 0.5] *= -1
        mi_data = sigmoid(ma_data @ W1) @ W2
        X = np.delete(X, m, 1)
        X = np.insert(X, m, mi_data.T, 1)

    return X


def generate_nonlinear_data(n, d, ee, graph_type, sem_type, seed=24):
    '''
    :param n: the number of samples
    :param d: the number of variables
    :param ee: edge density
    :param graph_type: ER or SF
    :param sem_type: gp or gp-add
    '''
    B = generate_graph(d=d, e=ee*d, graph_type=graph_type, seed=seed)
    X = simulate_data(B=B, n=n, sem_type=sem_type)
    return B, X

def generate_Mg_data(n, d, ee, ma_num, mi_num, graph_type, sem_type):
    '''
    :param n: the number of samples
    :param d: the number of variables
    :param ee: edge density
    :param ma_num: the number of macro-variables
    :param mi_num: the number of micro-variables of each macro-variable
    :param graph_type: ER or SF
    :param sem_type: gp or gp-add
    '''
    B = generate_graph(d=d, e=ee*d, graph_type=graph_type)
    Ma_B, Mi_B = get_Mg_graph(B=B, ma_num=ma_num, mi_num=mi_num)
    while Ma_B.shape[0] != d+ma_num:
        B = generate_graph(d=d, e=ee * d, graph_type=graph_type)
        Ma_B, Mi_B = get_Mg_graph(B=B, ma_num=ma_num, mi_num=mi_num)
    X = simulate_data(B=Ma_B, n=n, sem_type=sem_type)
    Mg_X = get_Mg_data(v=d+ma_num, X=X, d=d, mi_num=mi_num)
    return Mi_B, Mg_X

if __name__ == '__main__':
    B, X = generate_nonlinear_data(n=1000, d=20, ee=2, graph_type='ER', sem_type='gp-add')