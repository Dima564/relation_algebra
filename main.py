# -*- coding: utf-8 -*-
__author__ = "Kovalenko Dima"
import numpy as np
import itertools as it

# Note that this code is not designed to be understandable nor readable
# My goal was to check relation properties in fewest lines possible.

# Section Properties
def reflexive(r):
    return all([r[i][i] for i in range(0, len(r))])

def irreflexive(r):
    return all([not r[i][i] for i in range(0, len(r))])

def coreflexive(r):
    return not any([any(r[i][:i]) or any(r[i][i+1:]) for i in range(0, len(r))])

def symmetric(r):
    return np.array_equal(r, r.transpose())

def asymmetric(r):
    return np.array_equal(r * r.transpose(), np.zeros((len(r), len(r)), dtype=int))

def antisymmetric(r):
    return all([all([not(r[i][j] and r[j][i]) for j in range(0, i)]) for i in range(0, len(r))])

# a ** b means "a is implied by b"
def transitive(r):
    return all([r[i][j] ** v for (i, j), v in np.ndenumerate(r.dot(r))])


def negative_transitive(r):
    return transitive(np.logical_not(r))

# Note that this implementation is based on acyclic definition and is not efficint. topological sort will be a better solution
# reduce(lambda l, r: (l + [l[-1].dot(r)]), range(0,len(r)), [r]) returns len(r) multiplications of r
# any((r.transpose() * rn).flatten())  searches if any of theese multiplications is not empty
def acyclic(r):
    return not any([any((r.transpose() * rn).flatten()) for rn in reduce(lambda l, r: (l + [l[-1].dot(r)]), range(0, len(r)), [r])])

def connectivity(r):
    return all([all([r[i][j] or r[j][i] for j in range(0, i+1)]) for i in range(0, len(r))])

def weak_connectivity(r):
    return all([all([r[i][j] or r[j][i] for j in range(0, i)]) for i in range(0, len(r))])


# Helper functions
def upper(r, idx):
    return set(r[:, idx].nonzero()[0])

def lower(r, idx):
    return set(r[idx, :].nonzero()[0])

def symmetric_part(r):
    return r * r.transpose()

def asymmetric_part(r):
    return r * np.logical_not(r.transpose())

def uncomaparable(r):
    return np.logical_not(r + r.transpose()).astype(int)

# Section factorization
def factor_set(eq_relation):
    return list(set([tuple(lower(eq_relation, i)) for i in range(0, len(eq_relation))]))

def factorize_relation(r, eq):
    fs = factor_set(eq)
    l = len(factor_set)
    fact_r = np.zeros(shape=(l, l))
    for (i, j), v in np.ndenumerate(fact_r):
        fact_r[i][j] = any(r[i][j] for i, j in it.product(fs[i], fs[j]))
    return fact_r

# Section domination

def largest_for_P(r):
    return set([i for i in range(0, len(r)) if lower(r, i) == set(range(0, i) + range(i+1, len(r)))])

def largest_for_R(r):
    return set([i for i in range(0, len(r)) if lower(r, i) == set(range(0, len(r)))])


def strict_largest_for_R(r):
    return set([i for i in range(0, len(r)) if upper(r, i) <= set([i]) and lower(r, i) == set(range(0, len(r)))])

# Section blocking

def max_for_P(r):
    return set([i for i in range(0, len(r)) if not upper(r, i)])


def max_for_R(r):
    return set([i for i in range(0, len(r)) if upper(r, i) <= lower(r, i)])


def strict_max_for_R(r):
    return set([i for i in range(0, len(r)) if upper(r, i) <= set([i])])


# Section Von Neumann–Morgenstern
def calc_C_sets(r):
    C = [set()]
    while C[-1] != set(range(0, len(r))):  # this condition might not be correct. maybe use C[-1] != C[-2] ?
        C.append(set([i for i in range(0, len(r)) if upper(r, i) - set([i]) <= C[-1]]))
    return C

def neuman_morgenstern(r):
    C = calc_C_sets(r)
    elements = np.hstack(map(lambda i: list(C[i] - C[i-1]), range(1, len(C))))
    return reduce(lambda s, i: s | set([i]) if not upper(r, i) & s else s, elements, set())

# Section properties
def properties(r):
    props = set()
    props.add("reflexive") if reflexive(r) else None
    props.add("irreflexive") if irreflexive(r) else None
    props.add("coreflexive") if coreflexive(r) else None
    props.add("symmetric") if symmetric(r) else None
    props.add("asymmetric") if asymmetric(r) else None
    props.add("antisymmetric") if antisymmetric(r) else None
    props.add("transitive") if transitive(r) else None
    props.add("negative_transitive") if negative_transitive(r) else None
    props.add("acyclic") if acyclic(r) else None
    props.add("connectivity") if connectivity(r) else None
    props.add("weak_connectivity") if weak_connectivity(r) else None
    return props


# Section relation classes
def equivalence(props=set()):
    return set(["reflexive", "symmetric", "transitive"]).issubset(props)

def strict_order(props=set()):
    return set(["irreflexive", "transitive"]).issubset(props)

def total_order(props=set()):
    return set(["reflexive", "antisymmetric", "transitive"]).issubset(props)

def pre_order(props=set()):  # quazi-order
    return set(["reflexive", "transitive"]).issubset(props)

def weak_ordering(props=set()):
    return set(["asymmetric", "negative_transitive"]).issubset(props)



def get_relation_info(r):
    props = properties(r)
    classes = set()
    classes.add("equivalence") if equivalence(props) else None
    classes.add("strict_order") if strict_order(props) else None
    classes.add("total_order") if total_order(props) else None
    classes.add("pre_order") if pre_order(props) else None
    classes.add("weak_ordering") if weak_ordering(props) else None
    return (props, classes)


def main():
    # eq = np.genfromtxt('factorize_example_eq.csv', delimiter=',').astype(bool)
    r = np.genfromtxt('42.csv', delimiter=',').astype(int)
    a = neuman_morgenstern(r)
    print a

    # nr = uncomaparable(r):

    # print get_relation_info(r)
    # print get_relation_info(asym_part)
    # print get_relation_info(sym_part)
    # print get_relation_info(nr)
    # print


main()
