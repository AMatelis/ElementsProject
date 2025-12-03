import numpy as np
from simulation_viewer import ReactionRuleEngine, Atom, estimate_bond_energy, covalent_radius, max_valence_guess


def mk_atom(uid, symbol, pos=(0.0,0.0)):
    a = Atom(uid, symbol, pos=np.array(pos))
    return a


def test_rule_h_o():
    engine = ReactionRuleEngine()
    a = mk_atom('a1', 'H', pos=(0.5,0.5))
    b = mk_atom('a2', 'O', pos=(0.505,0.5))
    should, score = engine.should_form_bond(a, b, None)
    assert should is True
    assert score > 0.0


def test_rule_na_cl():
    engine = ReactionRuleEngine()
    a = mk_atom('n1', 'Na', pos=(0.2, 0.2))
    b = mk_atom('c1', 'Cl', pos=(0.201, 0.2))
    should, score = engine.should_form_bond(a, b, None)
    assert should is True
    assert score > 0.0


def test_rule_c_h():
    engine = ReactionRuleEngine()
    c = mk_atom('c1', 'C', pos=(0.4,0.4))
    h1 = mk_atom('h1', 'H', pos=(0.401,0.4))
    # C with no bonds should allow multiple H attachments, score positive
    should, score = engine.should_form_bond(c, h1, None)
    assert should is True
    assert score > 0.0


def test_rule_c_o():
    engine = ReactionRuleEngine()
    c = mk_atom('c1', 'C', pos=(0.6,0.6))
    o = mk_atom('o1', 'O', pos=(0.601,0.6))
    should, score = engine.should_form_bond(c, o, None)
    assert should is True
    assert score > 0.0
