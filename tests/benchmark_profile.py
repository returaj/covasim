'''
Benchmark the simulation
'''

import sciris as sc
import covasim as cv
from test_baselines import make_sim

sim = make_sim(use_defaults=False, do_plot=False) # Use the same sim as from the regression/benchmarking tests
to_profile = 'step' # Must be one of the options listed below

func_options = {
    'make_contacts': cv.make_random_contacts,
    'make_randpop':  cv.make_randpop,
    'person':        cv.Person.__init__,
    'make_people':   cv.make_people,
    'init_people':   sim.init_people,
    'initialize':    sim.initialize,
    'run':           sim.run,
    'step':          sim.step,
    'infect':        cv.People.infect,
}

sc.profile(run=sim.run, follow=func_options[to_profile])
