import sympy as sp
import re
import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
from evaluation import precision, recall, accuracy, f_beta_score, Metrics
from tqdm import tqdm

class DataStreamInterface(ABC):

    @abstractmethod
    def get_item(self):
        pass
    
    @abstractmethod
    def get_size(self):
        pass

class DataStream(DataStreamInterface):
    def __init__(self, csv_file, timestamp_col, id_col):
        self.data = pd.read_csv(csv_file)
        self.data = self.data.drop(columns=[id_col])
        # convert timestamp column
        self.data[timestamp_col] = pd.to_datetime(
            self.data[timestamp_col]
        )

        # set timestamp as index
        self.data = self.data.set_index(timestamp_col)

        # pointer to current row
        self.index = 0

    def get_item(self):
        """
        Returns next row in the stream.
        """
        if self.index >= len(self.data):
            raise StopIteration("End of stream reached")

        row = self.data.iloc[self.index]
        timestamp = self.data.index[self.index]
        self.index += 1

        return timestamp, row
    
    def get_size(self):
        return len(self.data)



def symbolic_not_xor(expr) -> bool:
    """
    Returns True if all superscript k values
    in a symbolic sum are identical.
    
    Example accepted symbols:
    delta_1_2^1
    alpha_3_4^2
    """
    
    # Get individual terms in the sum
    terms = expr.as_ordered_terms()
    
    k_values = []
    
    for term in terms:
        symbol_name = str(term)
        
        # Extract value after ^
        match = re.search(r"\^(\d+)", symbol_name)
        
        if match:
            k_values.append(match.group(1))
    
    # No valid terms found
    if not k_values:
        return 0

    
    # Check if all k values are identical
    return float(len(set(k_values)) == 1)




def check_common_i(expr) -> bool:
    """
    Returns True if all terms share one common intermediate index i:
    
    valid forms:
        delta_l_i^k
        delta_i_j^k
    
    Coefficients (e.g. 1.0*) are ignored.
    """

    terms = expr.as_ordered_terms()
    pairs = []

    for term in terms:
        
        # Remove numerical coefficient if present
        coeff, symbolic_part = term.as_coeff_Mul()
        
        term_str = str(symbolic_part)

        # search instead of match (more robust)
        match = re.search(r"[a-zA-Z]+_(\d+)_(\d+)\^\d+", term_str)

        if not match:
            return 0

        first_idx = int(match.group(1))
        second_idx = int(match.group(2))

        pairs.append((first_idx, second_idx))

    # possible common i candidates
    possible_i = set()

    for a, b in pairs:
        possible_i.add(a)
        possible_i.add(b)

    for i in possible_i:
        valid = 1

        for a, b in pairs:
            if a != i and b != i:
                valid = 0
                break

        if valid:
            return 1

    return 0


def apply_symbolic_not_xor_matrix(M):
    """
    Apply symbolic_not_xor() to every cell of matrix M.
    
    Returns:
        Matrix of booleans
    """
    
    rows, cols = M.shape
    
    result = sp.Matrix(rows, cols, lambda i, j: False)
    
    for i in range(rows):
        for j in range(cols):
            expr = M[i, j]
            result[i, j] = symbolic_not_xor(expr)
    
    return result

def get_delta(i, j, k):
    key = (i, j, k)
    if key not in delta_sym:
        # Nom pour affichage : \delta_{i,j}^{(k)}
        name = r'delta_%d_%d^%d' % (i, j, k)
        delta_sym[key] = sp.Symbol(name)
    return delta_sym[key]

def apply_check_common_i_matrix(M):
    """
    Apply check_common_i() to every cell of matrix M.
    
    Returns:
        Matrix of booleans
    """
    
    rows, cols = M.shape
    
    result = sp.Matrix(rows, cols, lambda i, j: False)
    
    for i in range(rows):
        for j in range(cols):
            expr = M[i, j]
            result[i, j] = check_common_i(expr)
    
    return result


def ones_M_ones(M, mode):
    """
    Computes:
    
        1^T * M * 1
    
    for a square SymPy matrix M.
    
    Returns:
        scalar
    """
    
    rows, cols = M.shape
    
    if rows != cols:
        raise ValueError("Matrix must be square")
    
    # column vector of ones
    ones_col = sp.ones(rows, 1)
    
    # row vector of ones
    ones_row = ones_col.T
    
    func = None
    if mode == "not_xor":
        func = apply_symbolic_not_xor_matrix
    elif mode == "common_index":
        func = apply_check_common_i_matrix

    result = ones_row * func(M * ones_col)
    
    return result[0]   # extract scalar

def get_incident_neighbors(M: sp.Matrix, enabled_states) -> int:
    """
    Returns the number of unique incident neighbors
    of all enabled states.

    Parameters
    ----------
    M : sp.Matrix
        Adjacency matrix
    enabled_states : iterable
        List/array of enabled state indices
        OR binary vector (0/1)

    Returns
    -------
    int
        Number of unique predecessor neighbors
    """

    # If binary vector is provided → convert to indices
    if isinstance(enabled_states, (list, np.ndarray)):
        enabled_states = np.array(enabled_states)



    incident_neighbors = set()
    
    for state in enabled_states:

        # scan incoming edges
        for i in range(M.rows):
            if M[i, state] != 0:
                incident_neighbors.add(i)

    return len(incident_neighbors)

def control_logic_verifier(X, S_1: np.array, S_2: np.array):
    """
    X   : square SymPy matrix
    S_1 : numpy binary vector
    S_2 : numpy binary vector

    """
    num_neighbors = get_incident_neighbors(X, np.where(S_2 == 1)[0])


    # Diagonal masks (kept in case needed later)
    D_s1 = sp.diag(*[float(s) for s in list(S_1)])
    D_s2 = sp.diag(*[float(s) for s in list(S_2)])

    # Matrix-vector multiplications
    E_1 = X*D_s1
    E_2 = X*D_s2

    # First verification
    if int(ones_M_ones(E_2, mode="not_xor")) != num_neighbors:
        return False

    # Second verification
    if ones_M_ones(E_1.T + E_2, mode="common_index") != num_neighbors and E_1 != E_2:
        return False

    return True

def control_logic_verifier_ones_M_ones(X, S: np.array):
    """
    X   : square SymPy matrix
    S_1 : numpy binary vector
    S_2 : numpy binary vector

    """
    num_neighbors = get_incident_neighbors(X, np.where(S == 1)[0])


    # Diagonal masks (kept in case needed later)
    D_s = sp.diag(*[float(s) for s in list(S)])

    # Matrix-vector multiplications
    E = X*D_s

    # First verification
    if int(ones_M_ones(E, mode="not_xor")) != num_neighbors:
        return False

    return True

def control_logic_verifier_common_index(X, S_1: np.array, S_2: np.array):
    """
    X   : square SymPy matrix
    S_1 : numpy binary vector
    S_2 : numpy binary vector

    """
    num_neighbors = get_incident_neighbors(X, np.where(S_2 == 1)[0])


    # Diagonal masks (kept in case needed later)
    D_s1 = sp.diag(*[float(s) for s in list(S_1)])
    D_s2 = sp.diag(*[float(s) for s in list(S_2)])

    # Matrix-vector multiplications
    E_1 = X*D_s1
    E_2 = X*D_s2
    
    # Second verification
    if ones_M_ones(E_1.T+E_2, mode="common_index") != num_neighbors and E_1 != E_2:
        return False

    return True

def control_logic_verifier_transition(X, S_1: np.array, S_2: np.array, convergence=None, details=False):
    """
    Verifies the transition S_1 -> S_2 with the fired-transition matrix

        Phi = diag(l) * X * diag(n),   l = S_1 o (1 - S_2)  (left states)
                                       n = S_2 o (1 - S_1)  (entered states)

    i.e. Phi = diag(l) * E_2 * diag(n). Persistent states (S_1 o S_2) do not appear in Phi.

    Conditions:
        (1) no deadlock:   1^T S_2 >= 1
        (2) justification: every entered state j has (1^T Phi)_j != 0
        (3) divergence:    for every left state i, (Phi 1)_i equals exactly one
                           group polynomial G_i^k = sum_{j : k(i,j) = k} delta_i_j^k
                           (AND: one group = all successors, OR: one alternative)
        (4) convergence:   for every entered state j, (1^T Phi)_j equals the sum of
                           X[p, j] over one of the predecessor groups of j

    X           : square SymPy matrix of delta_i_j^k symbols
    S_1, S_2    : binary vectors at t-1 and t
    convergence : {j: [set of predecessors, ...]} groups that may enter j together.
                  States not listed are OR-joins (each predecessor is its own group).
    details     : if True, returns (valid, dict) with Phi, sums and the violations
    """
    n_states = X.rows
    S_1 = np.asarray(S_1, dtype=int)
    S_2 = np.asarray(S_2, dtype=int)
    convergence = convergence or {}

    left = S_1 * (1 - S_2)
    entered = S_2 * (1 - S_1)

    # Fired-transition matrix
    Phi = sp.diag(*[int(s) for s in left]) * X * sp.diag(*[int(s) for s in entered])
    row_sums = Phi * sp.ones(n_states, 1)
    col_sums = sp.ones(1, n_states) * Phi

    errors = []

    # (1) no deadlock
    if S_2.sum() == 0:
        errors.append("deadlock: no state enabled")

    # (2) every entered state comes from a fired transition
    for j in np.where(entered == 1)[0]:
        if col_sums[j] == 0:
            errors.append(f"state {j} entered without a fired transition")

    # (3) divergence: each left state fires exactly one whole k group
    for i in np.where(left == 1)[0]:
        groups = {}
        for j in range(n_states):
            if X[i, j] != 0:
                k = re.search(r"\^(\d+)", str(X[i, j])).group(1)
                groups[k] = groups.get(k, 0) + X[i, j]
        if not groups:          # sink state: may terminate
            continue
        matches = [k for k, g in groups.items() if row_sums[i] == g]
        if len(matches) != 1:
            errors.append(f"state {i} left without firing exactly one whole divergence group")

    # (4) convergence: each entered state is reached by exactly one predecessor group
    for j in np.where(entered == 1)[0]:
        if col_sums[j] == 0:
            continue            # already reported by (2)
        preds = [p for p in range(n_states) if X[p, j] != 0]
        groups = convergence.get(int(j), [{p} for p in preds])
        polys = [sum(X[p, j] for p in g) for g in groups]
        if col_sums[j] not in polys:
            errors.append(f"state {j} entered by an invalid set of predecessors")

    valid = not errors
    if details:
        return valid, {
            "Phi": Phi,
            "left": left,
            "entered": entered,
            "row_sums": row_sums,
            "col_sums": col_sums,
            "errors": errors,
        }
    return valid

def control_logic(X, ds: DataStreamInterface, labels_to_exclude: list, beta: float):
    timestamp, S_1 = ds.get_item()
    tps = 0
    tns = 0
    fps = 0
    fns = 0

    for i in tqdm(range(ds.get_size()), desc="Processing dataset"):
        try:
            timestamp, S_2 = ds.get_item()
            if not control_logic_verifier(X, S_1, S_2):
                print("Error")
                # Update S_1 for the next iteration
            S_1 = S_2
            
    
        except StopIteration:
            print("End of data stream reached.")
            break
    return Metrics(fps=fps, 
                   fns=fns, 
                   tps=tps, 
                   tns=tns, 
                   precision=precision(tps, fps),
                   recall=recall(tps, fns),
                   accuracy=accuracy(tps, tns, fps, fns),
                   f_beta_score=f_beta_score(tps, fps, fns, beta))    

# Création des symboles δ_{i,j}^{(k)} (un seul exemplaire par triplet)
delta_sym = {}


# Initialisation d'une matrice 24×24 remplie de 0
M = sp.zeros(26, 26)

# ----- Remplissage des éléments non nuls -----
# Ligne 0
M[0,1] = get_delta(0,1,1)
M[0,6] = get_delta(0,6,1)

# Ligne 1
M[1,2] = get_delta(1,2,1)


# Ligne 2
M[2,3] = get_delta(2,3,1)



# Ligne 3
M[3,4] = get_delta(3,4,1)


# Ligne 4
M[4,5] = get_delta(4,5,1)


# Ligne 5
M[5,17] = get_delta(5,17,1)


# Ligne 6
M[6,7] = get_delta(6,7,1)
M[6,10] = get_delta(6,10,1)


# Ligne 7
M[7,8] = get_delta(7,8,1)


# Ligne 8
M[8,9] = get_delta(8,9,1)


# Ligne 9
M[9,3] = get_delta(9,3,1)
M[9,5] = get_delta(9,5,2)


# Ligne 10
M[10,11] = get_delta(10,11,1)
M[10,15] = get_delta(10,15,2)


# Ligne 11
M[11,9] = get_delta(11,9,1)
M[11,12] = get_delta(11,12,1)


# Ligne 12
M[12,13] = get_delta(12,13,1)


# Ligne 13
M[13,14] = get_delta(13,14,1)


# Ligne 14
M[14,5] = get_delta(14,5,1)


# Ligne 15
M[15,16] = get_delta(15,16,1)


# Ligne 16
M[16,11] = get_delta(16,11,1)


# Ligne 17
M[17,18] = get_delta(17,18,1)


# Ligne 18
M[18,19] = get_delta(18,19,1)
M[18,20] = get_delta(18,20,1)
M[18,21] = get_delta(18,21,1)


# Ligne 19
M[19,22] = get_delta(19,22,1)

# Ligne 20
M[20,23] = get_delta(20,23,1)


# Ligne 21
M[21,24] = get_delta(21,24,1)


# Ligne 22
M[22,25] = get_delta(22,25,1)


# Ligne 23
M[23,25] = get_delta(23,25,1)

# Ligne 24
M[24,25] = get_delta(24,25,1)

# Ligne 25
M[25,18] = get_delta(25,18,1)
M[25,0] = get_delta(25,0,2)

t2 = 1.0
t6 = 3.0
t7 = 3.0
t8 = 4.0

t11 = 5.0
t12 = 6.0
t13 = 6.0
t14 = 7.0
t20 = 10.0
t22 = 7.0
t23 = 5.0
t24 = 12.0
t9 = t12+t13+t14

t0 = 0.0
t1 = t6+t7+t8+t9-t2
t3 = 0.0
t4 = 0.0
t5 = 6.0

t10 = t7+t8-t11

t15 = 0.0
t16 = 0.0
t17 = 8.0
t18 = 9.0
t19 = t20+t23-t22
t21 = t20+t23-t24
t25 = 6.0


durations = {
    0: t1,
    1: t1,
    2: t2,
    3: t3,
    4: t4,
    5: t5,
    6: t6,
    7: t7,
    8: t8,
    9: t9,
    10: t10,
    11: t11,
    12: t12,
    13: t13,
    14: t14,
    15: t15,
    16: t16,
    17: t17,
    18: t18,
    19: t19,
    20: t20,
    21: t21,
    22: t22,
    23: t23,
    24: t24,
    25: t25
}

# Convergence groups: for each join state j, the sets of predecessors that enter j together
# (one set = AND-join, one singleton per predecessor = OR-join). Used by
# control_logic_verifier_transition; states not listed are treated as OR-joins.
convergence = {
    3: [{2, 9}, {2}],            # 9 chose 3 (k=1): wait for 2 and 9; 9 chose 5 (k=2): 2 alone
    5: [{4, 14}, {4, 9, 14}],    # 9 chose 3: 4 and 14; 9 chose 5: 4, 9 and 14
    9: [{8, 11}],
    11: [{10}, {16}],
    18: [{17}, {25}],
    25: [{22, 23, 24}],
}
# df = pd.read_csv("data.csv")
# ds = DataStream("data.csv", timestamp_col="timestamp", id_col="id")
# control_logic(X=M, ds=ds, labels_to_exclude=[], beta=1.0)
