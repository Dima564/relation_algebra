# -*- coding: utf-8 -*-
__author__ = "Kovalenko Dima"
import numpy as np
import itertools as it
import sys

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

def uncomaparable_part(r):
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
    return {i for i in range(0, len(r)) if upper(r, i) <= {i} and lower(r, i) == set(range(0, len(r)))}

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
    return reduce(lambda s, i: s | {i} if not upper(r, i) & s else s, elements, set())


# Section k-optimization

# Vectorized nonzero
def vec_nonzero(a):
    return [set(ai.nonzero()[0]) for i, ai in enumerate(a)]

# returns list of sets of element indexes, every ith set correspind to Hr0(i)
def Hr0(r):
    return vec_nonzero(r * np.logical_not(r).transpose() * np.logical_not(np.eye(len(r))))

def Er(r):
    return vec_nonzero(r * r.transpose() + np.eye(len(r)))

def Nr(r):
    return vec_nonzero(np.logical_not(r) * np.logical_not(r).transpose() * np.logical_not(np.eye(len(r))))

def get_Sk(r):
    hr0 = Hr0(r)
    er = Er(r)
    nr = Nr(r)
    # Meh. Should've used Matlab with it's vectorization
    return [[hr0[i] | er[i] | nr[i], hr0[i] | nr[i], hr0[i] | er[i], hr0[i]] for i in range(0, len(r))]

# Ski is an ith column of Sk
def get_i_max(Ski):
    a = set()
    for i, ith_row in enumerate(Ski):
        for j, jth_row in enumerate(Ski):
            if ith_row < jth_row:
                break
        else:
            a.add(i)
    return a

def get_k_max(Sk):
    return [get_i_max(Sk[:, i]) for i in range(0, 4)]


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
    return {"reflexive", "symmetric", "transitive"} <= props

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


def print_sk(Sk):
    row_format = "{x[0]:30}{x[1]:30}{x[2]:30}{x[3]:30}"
    for i, row in enumerate(Sk):
        print row_format.format(x=map(pretty_print_set, row))

def pretty_print_set(s):
    return "{" + ", ".join(str(v) for v in list(s)) + "}"

def pp(s):
    return pretty_print_set(s)

def main(filename):

    r = np.genfromtxt(filename, delimiter=',').astype(int)

    print "Relation: \n", r
    props, classes = get_relation_info(r)
    print "Properties: %s" % pretty_print_set(props)
    print "Classes: %s" % pretty_print_set(classes)

    sym, asym, uc = symmetric_part(r), asymmetric_part(r), uncomaparable_part(r)
    p_asym, c_asym = get_relation_info(asym)
    p_sym, c_sym = get_relation_info(sym)
    p_uc, c_uc = get_relation_info(uc)
    print "\nAsymmetric part:\n", asym
    print "Properties: %s" % pretty_print_set(p_asym)
    print "Classes: %s" % pretty_print_set(c_asym)
    print "\nSymmetric part:\n", sym
    print "Properties: %s" % pretty_print_set(p_sym)
    print "Classes: %s" % pretty_print_set(c_sym)
    print "\nUncomparable part:\n", uc
    print "Properties: %s" % pretty_print_set(p_uc)
    print "Classes: %s" % pretty_print_set(c_uc)

    print
    print "Largest for R:", pp(largest_for_R(r))
    print "Strict Largest for R:", pp(strict_largest_for_R(r))
    print "Largest for P:", pp(largest_for_P(r))

    print
    print "Max for R:", pp(max_for_R(r))
    print "Strict Max for R:", pp(strict_max_for_R(r))
    print "Max for P:", pp(max_for_P(r))

    print
    print "Neuman-Morgenstern solution:\n" + pp(neuman_morgenstern(r))
    a = np.array(get_Sk(r))
    print "\nSk table:"
    print_sk(a)
    print "\nSets of K-max elements:"
    print [pp(x) for x in get_k_max(a)]

if len(sys.argv) != 2:
    print "Usage: python main.py data.csv"
    main("53.csv")
else:
    main(sys.argv[1])
