from functools import partial

from .al_pinn import PointSelector

# from .gd import GDPointSelector
from .eigval import EigenvaluePointSelector
from .random import RandomPointSelector
from .residue import ResidueSelector


AL_CONSTRUCTOR = {
    # 'gd': GDPointSelector,
    'eig_greedy': partial(EigenvaluePointSelector, selection_method='greedy'),
    'eig_sampling': partial(EigenvaluePointSelector, selection_method='sampling'),
    'eig_kmeans': partial(EigenvaluePointSelector, selection_method='kmeans'),
    'random': RandomPointSelector,
    'residue': ResidueSelector,
}
