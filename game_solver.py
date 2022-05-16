import numpy as np
from scipy import optimize
import itertools


class Game:
    """Nash Equilibrium Solver"""

    def __init__(self, U, label="a strategic form game"):
        """Create a strategic form game with a given payoff matrix
        U: the payoff matrix of an n-player game
        (dim=n+1, shape=|A0|*...*|An-1|*n), U[a0,...,an-1,i] is the payoff of player i 
        under the action profile (a0,...,an-1)
        """
        self.U = U
        self.label = label

    @property
    def U(self):
        """Property getter for payoff matrix U"""
        return self.__U

    @U.setter
    def U(self, U):
        """Property setter for payoff matrix U"""
        self.__U = U
        self.n = U.ndim - 1  # number of players
        # action profile A=[A0,...,An-1], where Ai = [0,1,...,|Ai|-1] is i's action set,
        # Ai[k] means the k-th action of i
        self.A = [list(range(Ai)) for Ai in U.shape[:-1]]  # action profile
        # self.nA = [|A0|,...,|An-1|]
        self.nA = list(
            U.shape[:-1]
        )  # the number of available actions for each player on A

    def parse(self, z):
        """parse list z to get a nested list of strategy profile and a value profile list.
        z: a list/array (len = sum(nA)+n), z=[p,v]=[p0,p1,...,pn,v],
        where pi is i's strategy profile vector, v is the (len=n) payoff vector
        """
        last_index = np.cumsum(
            self.nA
        )  # the indexes for each player's last action if we ravel A
        p_raw = z[: np.sum(self.nA)]  # p_raw is (len=nA) list for strategy profile
        v = z[np.sum(self.nA) :]  # v is (len=n) list for value profile in NE
        # rearrange p_raw to use a nested list p to represent each player's strategy profile
        p = []  # p is the nested list of strategy profile
        for i in range(self.n):  # loop for each player
            start = (
                last_index[i - 1] if i >= 1 else 0
            )  # index of first action of i if we ravel A
            end = last_index[i]  # index of last action of i if we ravel A
            pi = p_raw[start:end]  # i's strategy profile
            p.append(pi)  # add i's strategy profile to p
        return p, v

    def inner(self, z, S):
        """write the system of equations for (potential) mixed NE on the support profile S
        z: 1dim array/list (len=sum(nA)+n) of strategy profile and value profile
        S: support profile (nested list) [S0,...,Sn-1], where Si=[ai0,...,aim] is i's support, m=|Si|-1
        """
        supports = list(itertools.product(*S))  # all the action profiles on S
        eqs = []  # a list of (sum(nA) + n) equations for NE
        p, v = self.parse(
            z
        )  # p is the nested list of strategy profile, v is value profile
        # 1. all actions outside support profile S have 0 probability
        for i in range(self.n):  # loop for each player
            if set(S[i]) < set(self.A[i]):  # if Si is a proper subset of Ai
                pi = p[i]  # pi is i's strategy profile
                # player i won't play any action j outside his support Si
                eqs += [pi[j] - 0 for j in set(self.A[i]) - set(S[i])]
        # 2. n equations from the definition of strategy profile
        eqs += [
            np.sum(pi) - 1 for pi in p
        ]  # for each player, sum of (probabilities of) strategies is 1
        # 3. construct the other sum_i{|S_i|} equations for the expected utility equivalence on suppport
        for i in range(self.n):  # loop for each player
            vi = v[i]  # i's expected utility in NE
            for ai in S[i]:  # loop for each action ai of i on Si
                utility = 0  # expected utility of i if i plays ai
                for s in supports:  # loop for all action profiles on S
                    if s[i] == ai:  # if player i plays ai in action profile s
                        # probability of s happens conditional on i plays ai
                        # i.e. prob(a_{-i}) = product(pj(aj)),j!=i
                        prob = np.product([p[j][s[j]] for j in range(self.n) if j != i])
                        u = self.U[
                            s + (i,)
                        ]  # utility of player i under action profile s
                        utility += prob * u  # update expected utility for i playing ai
                eqs.append(
                    utility - vi
                )  # NE requires all the actions on Si has same expected utility vi
        return eqs  # return the system of equations required by NE

    def outer(self, p, v, S):
        """check whether the strategy and value profile got from solving equations on support profile S is NE
        p: [p0,...,pn-1] is the nested list of strategy profile, where pi is i's strategy vector
        v: (len=n) list of value profile
        S: support profile (nested list) [S0,...,Sn-1], where Si=[ai0,...,aim] is i's support, m=|Si|-1
        """
        # 1. check if there is any p_im (the prob of player i using action m) < 0 on S
        for pi in p:
            # we treat very small negative float number like -1e-20 returned
            # by scipy.optimize.root in inner(z, S) as 0
            if np.any(np.array(pi) < (-1e-6)):
                return (
                    False  # failure if some player plays an action with negative prob
                )
        # 2. check if every action on support Si is **no worse** than action outside Si for each player i
        # outside: all the action profiles of which **only one** player's action is outside the support S
        outside = []
        if self.A != S:  # if support profile is smaller than action profile
            for i in range(self.n):  # loop for each player
                if self.A[i] != S[i]:  # if i's support is smaller than his action set
                    # Anew is the (nested list) of action profile when i plays some action not on Si,
                    # while all the other plays stay on S(-i)
                    Anew = (
                        S[:i] + [list(set(self.A[i]) - set(S[i]))] + S[i + 1 :]
                    )  # nested list
                    outside += list(itertools.product(*Anew))  # update outside
        # check the validity of NE if support profile is smaller than action profile
        if outside:  # if outside is not empty
            for i in range(self.n):  # loop for each player
                if set(self.A[i]) - set(
                    S[i]
                ):  # if player i can take some action not on Si
                    vi = v[i]  # i's expected utility on S
                    for ai in set(self.A[i]) - set(
                        S[i]
                    ):  # loop for each action ai of i but not on Si
                        utility = 0  # expected utility if i plays ai
                        for (
                            s
                        ) in (
                            outside
                        ):  # loop for all action profiles (tuple s) outside S
                            if s[i] == ai:  # if player i plays ai in action profile s
                                # probability of s happens conditional on i plays ai (ai not in Si)
                                # prob(a_{-i}) = product(pj(aj)),j!=i
                                prob = np.product(
                                    [p[j][s[j]] for j in range(self.n) if j != i]
                                )
                                u = self.U[
                                    s + (i,)
                                ]  # utility of i under action profile s
                                utility += (
                                    prob * u
                                )  # update expected utility for i playing ai not in Si
                        # failure if i playing any ai not in Si can improve his expected utility
                        if utility - vi > 1e-6:
                            return False
        return True  # all the checks have passed

    def feasibility(self, S, method="hybr"):
        """find the mixed NE in n-player game given supports S=(S0, ..., Sn-1)
        S: support profile (nested list) [S0,...,Sn-1], 
        where Si=[ai0,...,aim] is i's support, ai1<...<aim, m=|Si|-1
        method: algorithm used in scipy solver
        """
        # 1.solve all the (n+sum(An)) equations for (n+sum(An)) unknowns
        try:
            f = lambda z: self.inner(z, S)  # solve the system of equations
            # we set each player plays each action with prob 0.5 and
            # each player's expected utility is 0 in initial guess
            sol = optimize.root(
                f, [0.5] * np.sum(self.nA) + [0] * self.n, method=method
            )
            x = sol.x
            x[
                np.abs(x) < 1e-6
            ] = 0.0  # change very small float number returned by scipy solver to 0
            x[: np.sum(self.nA)][
                (1 - 1e-6) < x[: np.sum(self.nA)]
            ] = 1.0  # treat prob very close to 1 as 1
            # if support is for mixed strategy profile but each player plays pure strategy
            if np.any(np.array([len(Si) for Si in S]) != 1) and set(
                x[: np.sum(self.nA)]
            ) == {0.0, 1.0}:
                return None  # redundant solution since we also can get it on the pure strategy support
            else:
                p, v = self.parse(x)
        except ValueError:  # if the system of equations cannot be solved
            return None  # no NE on the support
        # 2.if the system of equations has solution, we check whether the solution is NE
        if self.outer(p, v, S) == True:
            return p, v  # return strategy profile and value profile for NE
        else:
            return None  # no NE on the support

    def powerset(self, iterable):
        "Find power set (excluding empty set) from an iterable object"
        s = list(iterable)
        return list(
            itertools.chain.from_iterable(
                itertools.combinations(s, r) for r in range(1, len(s) + 1)
            )
        )

    def size_profile(self):
        """Find all possible size profiles in the game"""
        if self.n == 2:
            # X is (|A0|*|A1|)*2 array, X[i] is the i-th size profile vector
            X = np.array(
                [
                    [x0, x1]
                    for x0 in range(1, self.nA[0] + 1)
                    for x1 in range(1, self.nA[1] + 1)
                ]
            )
            balance = np.abs(
                np.diff(X).ravel()
            )  # balance of support profile: abs(x0-x1)
            size = np.sum(X, axis=1)  # size of support profile
            # first sort on balance, second sort on size
            # https://numpy.org/doc/stable/reference/generated/numpy.lexsort.html
            ind = np.lexsort((size, balance))  # balance is the primary sort key
            X = X[ind]
            return X
        elif self.n > 2:
            # nested list of each player's action set
            nest = [list(range(1, self.nA[i] + 1)) for i in range(self.n)]
            # X is (|A0|*...*|An-1|)*n array, X[i] is the i-th size profile vector
            X = np.array(list(itertools.product(*nest)))
            balance = np.max(X, axis=1) - np.min(
                X, axis=1
            )  # balance of support profile: max_{i,j}(xi - xj)
            size = np.sum(X, axis=1)  # size of support profile
            # first sort on size, second sort on balance (different for two players case!!)
            ind = np.lexsort((balance, size))  # size is the primary sort key
            X = X[ind]
            return X

    def ne(self, only_one=False, method="hybr"):
        """Find all the (pure and mixed) NEs (if only_one==False)
        return strategy profiles + value profiles
        only_one: only find one sample NE if True
        method: algorithm used in scipy solver
        """
        self.X = self.size_profile()  # rows of X are all the size profiles in the game
        if self.n == 2:  # 2-player algorithm
            NE1 = self.algorithm1(only_one, method)
            return NE1
        elif self.n > 2:  # n-player algorithm
            NEs = self.algorithm2(only_one, method)
            return NEs
        else:
            print("The number of players must be greater than or equal to 2")

    def dominated(self, i, a, Ai, R):
        """check if an action a of player i in action set Ai
        is conditionally dominated given other players play actions on R 
        i: index of player i
        a: scalar of player i's action
        Ai: list of i's action set
        R: nested list other players' action profile (a subset of A(-i))
        """
        for ai in set(Ai) - {
            a,
        }:  # loop for all i's actions except a
            for (
                b
            ) in (
                R
            ):  # loop for all others's action profile, b=[b0,.., bi-1, bi+1,...,bn]
                # if playing *ai* cannot increase i's utility compared to *a* when others play b
                if (
                    self.U[tuple(b[:i]) + (a,) + tuple(b[i:]) + (i,)]
                    >= self.U[tuple(b[:i]) + (ai,) + tuple(b[i:]) + (i,)]
                ):
                    break  # a is not strictly conditionally dominated by ai
            else:
                return True  # a is strictly conditionally dominated by ai
        else:
            return False  # a is not conditionally dominated by any action in Ai

    def algorithm1(self, only_one, method="hybr"):
        """Find all NEs in 2-player game if only_one == False
        only_one: only return one sample NE if True
        method: algorithm used in scipy solver
        """
        NE = []  # list of nash equilibriums
        A0, A1 = self.A  # player 0's and player 1's action set
        all_S0 = self.powerset(A0)  # all candidate supports for player 0
        for x in self.X:  # loop for all size profile x (X has been sorted)
            x0, x1 = x  # player 0's and player 1's support size
            for S0 in [
                S0 for S0 in all_S0 if len(S0) == x0
            ]:  # loop for each S0 which has the size of x0
                # A1new is all the actions in A1 which are not conditionally dominated, given S0
                # make sure R=[[s0] for s0 in S0] is a nested list
                A1new = [
                    a1
                    for a1 in A1
                    if not self.dominated(1, a1, A1, [[a0] for a0 in S0])
                ]
                # if there is *not* any a0 in S0 which is conditionally dominated, given A1new
                if not np.any(
                    [self.dominated(0, a0, S0, [[a1] for a1 in A1new]) for a0 in S0]
                ):
                    # loop for each subset (S1) of A1new which has the size of x1
                    for S1 in [S1 for S1 in self.powerset(A1new) if len(S1) == x1]:
                        # if there is **not** any a0 in S0 which is conditionally dominated, given S1
                        if not np.any(
                            [
                                self.dominated(0, a0, S0, [[a1] for a1 in S1])
                                for a0 in S0
                            ]
                        ):
                            try:  # try to find NE on support S
                                S = [list(S0), list(S1)]
                                (p, v) = self.feasibility(S, method=method)
                                if only_one:  # if only_one==True
                                    return (
                                        p,
                                        v,
                                    )  # return the first NE and break the program
                                NE.append((p, v))  # append the solution to NE list
                            except TypeError:  # self.feasibility returns None
                                pass  # no NE on given support (S0, S1)
        if NE:
            return NE  # if NE is not empty, return the list of all NEs
        else:
            print(
                "no NE, there are bugs!"
            )  # finite n person game must has at least one NE

    def algorithm2(self, only_one, method="hybr"):
        """Find all NEs in n-player game if only_one == False else only one sample NE
        only_one: only return one sample NE if True
        method: algorithm used in scipy solver
        """
        NEs = []  # list of nash equilibriums
        # S_candidates[i] is the list of candidates for player i's support
        S_candidates = [self.powerset(self.A[i]) for i in range(self.n)]
        for x in self.X:  # loop for all size profile x (X has been sorted)
            S = [[] for i in range(self.n)]  # un-instantiated support profile
            # domain profile (nested list), D[i] is the list of i's supports which have size of x[i]
            D = [
                [Si for Si in S_candidates[i] if len(Si) == x[i]] for i in range(self.n)
            ]
            if only_one == True:
                try:
                    p, v = self.rb(S, D, 0, only_one=True, method=method, NE=[])
                    return p, v  # return the first NE and break the program
                except TypeError:  # self.rb returns False or None (no NE on D)
                    pass
            else:  # update NE list if there are NEs returned by self.rb
                result = self.rb(S, D, 0, only_one=False, method=method, NE=[])
                if result:
                    NEs += result
        # return all NEs (delete all duplicates) if only_one=False
        if only_one == False:
            return NEs

    def rb(self, S, D, i, only_one, method, NE=[]):
        """
        Use recursive-backtracking method to find NEs given support profile S and domain profile D
        S: support profile
        D: domain profile (nested list), D[i] is the list of i's supports
        i: index of *next* support to instantiate
        only_one: find one sample NE if True, all NEs if False
        method: algorithm used in scipy solver
        NE: the list of nash equilibriums which have been found (useful if only_one=False)
        """
        if i == self.n:  # all Si in S have been filled
            try:  # if there is NE on S
                p, v = self.feasibility(S, method)
                if only_one:  # if only_one=True, return first NE and stop the program
                    return p, v
                else:  # if only_one=False, update NE list and continue to find other NEs
                    NE.append((p, v))
            except TypeError:  # if self.feasibility return None (no NE on S)
                pass
        else:  # if some Si are still empty
            for di in D[i]:  # loop for each candidate di for i's support Si
                S[i] = di  # instantiate Si
                # delete di from the candidate list for i's support since we have considered di to be Si
                # D[i] = [Si for Si in D[i] if di!=Si] # Is this necessary? or redundant?
                Dnew = [[S[k]] for k in range(i + 1)] + D[
                    i + 1 :
                ]  # use [S[k]] to replace D[k] for k=0,1,...,i
                Dnew = self.irsds(
                    Dnew
                )  # self.irsds(D) update D if D is valid else return False
                if Dnew:  # if domain profile D is reasonable
                    try:  # call self.rb again until return NE or False
                        if (
                            only_one == True
                        ):  # call self.rb recursively until finding the first NE
                            p, v = self.rb(
                                S, D=Dnew, i=i + 1, only_one=True, method=method, NE=[]
                            )
                            return p, v  # stop the loop
                        else:  # if only_one=False, call self.rb recursively to find all NEs
                            NE = self.rb(
                                S, D=Dnew, i=i + 1, only_one=False, method=method, NE=NE
                            )
                    except TypeError:
                        pass
        if only_one == False:
            return (
                NE  # if only_one=False, return all the NEs under the initial (S, D, i)
            )

    def irsds(self, D):
        """Iterated removal of strictly sominated strategies (and check if D is a valid domain profile)
        D: domain profile (nested list), D[i] is the list of i's supports
        """
        changed = True  # whether D has been shrunk
        while changed:  # if D changed
            changed = False
            for i in range(self.n):  # loop for all players
                # loop for all action ai on every Si of D[i]
                for ai in set(itertools.chain.from_iterable(D[i])):
                    Dothers = D[:i] + D[i + 1 :]  # D(-i)
                    for j in range(
                        len(Dothers)
                    ):  # D(-i)[j] is all the actions player j can take in D
                        Dothers[j] = set(itertools.chain.from_iterable(Dothers[j]))
                    # R: nested list of other players' action profile
                    # i.e., the union set of S(-i) for all all S(-i) in D(-i)
                    R = list(itertools.product(*Dothers))
                    if self.dominated(
                        i, ai, self.A[i], R
                    ):  # if ai is conditionally dominated, given R
                        D[i] = [
                            Si for Si in D[i] if ai not in Si
                        ]  # shrink D since i won't play ai
                        changed = True
                        if not D[i]:
                            return False  # D is invalid if D[i] is empty
        return D  # return the updated D (which has been shrunk)

