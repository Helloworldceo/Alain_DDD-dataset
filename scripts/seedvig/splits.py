"""
Evaluation splits shared by all SEED-VIG experiments.

lopo_folds(person)
    Leave-one-person-out: 21 folds; both sessions of persons 4 and 5 are held
    out together, so no person appears in training and test at the same time.

session_block_folds(session, t, k=5)
    Within-session protocol of the dataset authors: each session is divided
    into k contiguous time blocks; fold j tests on block j of every session and
    trains on the remaining blocks of the same sessions.
"""
import numpy as np


def lopo_folds(person):
    person = np.asarray(person)
    for p in sorted(np.unique(person), key=int):
        te = np.where(person == p)[0]
        tr = np.where(person != p)[0]
        yield p, tr, te


def session_block_folds(session, t, k=5):
    session, t = np.asarray(session), np.asarray(t)
    block = np.zeros(len(t), dtype=int)
    for s in np.unique(session):
        idx = np.where(session == s)[0]
        order = idx[np.argsort(t[idx])]
        for j, part in enumerate(np.array_split(order, k)):
            block[part] = j
    for j in range(k):
        yield j, np.where(block != j)[0], np.where(block == j)[0]
