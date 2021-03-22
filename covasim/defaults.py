'''
Set the defaults across each of the different files.

To change the default precision from 32 bit (default) to 64 bit, use::

    cv.options.set(precision=64)
'''

import numpy as np
import numba as nb
import sciris as sc
from .settings import options as cvo # To set options

# Specify all externally visible functions this file defines -- other things are available as e.g. cv.defaults.default_int
__all__ = ['default_float', 'default_int', 'get_colors', 'get_sim_plots', 'get_scen_plots']


#%% Specify what data types to use

result_float = np.float64 # Always use float64 for results, for simplicity
if cvo.precision == 32:
    default_float = np.float32
    default_int   = np.int32
    nbfloat       = nb.float32
    nbint         = nb.int32
elif cvo.precision == 64: # pragma: no cover
    default_float = np.float64
    default_int   = np.int64
    nbfloat       = nb.float64
    nbint         = nb.int64
else:
    raise NotImplementedError(f'Precision must be either 32 bit or 64 bit, not {cvo.precision}')


#%% Define all properties of people

class PeopleMeta(sc.prettyobj):
    ''' For storing all the keys relating to a person and people '''

    # Set the properties of a person
    person = [
        'uid',              # Int
        'age',              # Float
        'sex',              # Float
        'symp_prob',        # Float
        'severe_prob',      # Float
        'crit_prob',        # Float
        'death_prob',       # Float
        'rel_trans',        # Float
        'rel_sus',          # Float
        'prior_symptoms',   # Float
#        'sus_imm',          # Float
#        'trans_imm',        # Float
#        'prog_imm',         # Float
        'vaccinations',     # Number of doses given per person
        'vaccine_source',   # index of vaccine that individual received
        'NAb',              # Current neutralization titre relative to convalescent plasma
    ]

    # Set the states that a person can be in: these are all booleans per person -- used in people.py
    states = [
        'susceptible',
        'exposed',
        'infectious',
        'symptomatic',
        'severe',
        'critical',
        'tested',
        'diagnosed',
        'dead',
        'known_contact',
        'quarantined',
        'vaccinated'
    ]

    strain_states = [
        'exposed_strain',
        'exposed_by_strain',
        'infectious_strain',
        'infectious_by_strain',
        'recovered_strain',
    ]

    # Set the dates various events took place: these are floats per person -- used in people.py
    dates = [f'date_{state}' for state in states] # Convert each state into a date
    dates.append('date_pos_test') # Store the date when a person tested which will come back positive
    dates.append('date_end_quarantine') # Store the date when a person comes out of quarantine
    dates.append('date_recovered') # Store the date when a person recovers
    dates.append('date_vaccinated') # Store the date when a person is vaccinated

    # Duration of different states: these are floats per person -- used in people.py
    durs = [
        'dur_exp2inf',
        'dur_inf2sym',
        'dur_sym2sev',
        'dur_sev2crit',
        'dur_disease',
    ]

    all_states = person + states + strain_states + dates + durs


#%% Define other defaults

# A subset of the above states are used for results
result_stocks = {
        'susceptible':          'Number susceptible',
        'exposed':              'Number exposed',
        'exposed_by_strain':    'Number exposed by strain',
        'infectious':           'Number infectious',
        'infectious_by_strain': 'Number infectious by strain',
        'symptomatic':          'Number symptomatic',
        'severe':               'Number of severe cases',
        'critical':             'Number of critical cases',
        'diagnosed':            'Number of confirmed cases',
        'quarantined':          'Number in quarantine',
        'vaccinated':           'Number of people vaccinated'
}

# The types of result that are counted as flows -- used in sim.py; value is the label suffix
result_flows = {'infections':           'infections',
                'reinfections':         'reinfections',
                'infections_by_strain': 'infections_by_strain',
                'infectious':           'infectious',
                'infectious_by_strain': 'infectious_by_strain',
                'tests':                'tests',
                'diagnoses':            'diagnoses',
                'recoveries':           'recoveries',
                'symptomatic':          'symptomatic cases',
                'severe':               'severe cases',
                'critical':             'critical cases',
                'deaths':               'deaths',
                'quarantined':          'quarantined people',
                'vaccinations':         'vaccinations',
                'vaccinated':           'vaccinated people'
}

# Define these here as well
new_result_flows = [f'new_{key}' for key in result_flows.keys()]
cum_result_flows = [f'cum_{key}' for key in result_flows.keys()]

# Parameters that can vary by strain (should be in list format)
strain_pars = ['rel_beta',
               'asymp_factor',
               'dur',
               'rel_symp_prob',
               'rel_severe_prob',
               'rel_crit_prob',
               'rel_death_prob',
]

# Immunity is broken down according to 3 axes, as listed here
immunity_axes = ['sus', 'trans', 'prog']

# Immunity protection also varies depending on your infection history
immunity_sources = [
    'asymptomatic',
    'mild',
    'severe',
]

# Default age data, based on Seattle 2018 census data -- used in population.py
default_age_data = np.array([
            [ 0,  4, 0.0605],
            [ 5,  9, 0.0607],
            [10, 14, 0.0566],
            [15, 19, 0.0557],
            [20, 24, 0.0612],
            [25, 29, 0.0843],
            [30, 34, 0.0848],
            [35, 39, 0.0764],
            [40, 44, 0.0697],
            [45, 49, 0.0701],
            [50, 54, 0.0681],
            [55, 59, 0.0653],
            [60, 64, 0.0591],
            [65, 69, 0.0453],
            [70, 74, 0.0312],
            [75, 79, 0.02016], # Calculated based on 0.0504 total for >=75
            [80, 84, 0.01344],
            [85, 89, 0.01008],
            [90, 99, 0.00672],
            ])


def get_colors():
    '''
    Specify plot colors -- used in sim.py.

    NB, includes duplicates since stocks and flows are named differently.
    '''
    c = sc.objdict()
    c.susceptible           = '#4d771e'
    c.exposed               = '#c78f65'
    c.exposed_by_strain     = '#c75649',
    c.infectious            = '#e45226'
    c.infectious_by_strain  = '#e45226'
    c.infections            = '#b62413'
    c.reinfections          = '#732e26'
    c.infections_by_strain  = '#b62413'
    c.tests                 = '#aaa8ff'
    c.diagnoses             = '#5f5cd2'
    c.diagnosed             = c.diagnoses
    c.quarantined           = '#5c399c'
    c.vaccinations          = '#5c399c'
    c.vaccinated            = '#5c399c'
    c.recoveries            = '#9e1149'
    c.symptomatic           = '#c1ad71'
    c.severe                = '#c1981d'
    c.critical              = '#b86113'
    c.deaths                = '#000000'
    c.dead                  = c.deaths
    c.default               = '#000000'
    return c


def get_strain_colors():
    '''
       Specify plot colors -- used in sim.py.

       NB, includes duplicates since stocks and flows are named differently.
       '''
    colors = ['#4d771e', '#c78f65', '#c75649', '#e45226', '#e45226', '#b62413', '#732e26', '#b62413']
    return colors



# Define the 'overview plots', i.e. the most useful set of plots to explore different aspects of a simulation
overview_plots = [
            'cum_infections',
            'cum_infections_by_strain',
            'cum_severe',
            'cum_critical',
            'cum_deaths',
            'cum_diagnoses',
            'new_infections',
            'new_infections_by_strain',
            'new_severe',
            'new_critical',
            'new_deaths',
            'new_diagnoses',
            'n_infectious',
            'n_infectious_by_strain',
            'n_severe',
            'n_critical',
            'n_susceptible',
            'new_tests',
            'n_symptomatic',
            'new_quarantined',
            'n_quarantined',
            'new_vaccinations',
            'new_vaccinated',
            'cum_vaccinated',
            'cum_vaccinations',
            'test_yield',
            'r_eff',
            ]


def get_sim_plots(which='default'):
    '''
    Specify which quantities to plot; used in sim.py.

    Args:
        which (str): either 'default' or 'overview'
    '''
    if which in [None, 'default']:
        plots = sc.odict({
                'Total counts': [
                    'cum_infections',
                    'n_infectious',
                    'cum_diagnoses',
                ],
                'Daily counts': [
                    'new_infections',
                    'new_diagnoses',
                ],
                'Health outcomes': [
                    'cum_severe',
                    'cum_critical',
                    'cum_deaths',
                ],
        })
    elif which == 'overview':
        plots = sc.dcp(overview_plots)
    elif which == 'seir':
        plots = sc.odict({
                'SEIR states': [
                    'n_susceptible',
                    'n_preinfectious',
                    'n_infectious',
                    'n_removed',
                ],
        })
    else: # pragma: no cover
        errormsg = f'The choice which="{which}" is not supported'
        raise ValueError(errormsg)
    return plots


def get_scen_plots(which='default'):
    ''' Default scenario plots -- used in run.py '''
    if which in [None, 'default']:
        plots = sc.odict({
            'Cumulative infections': [
                'cum_infections',
            ],
            'New infections per day': [
                'new_infections',
            ],
            'Cumulative deaths': [
                'cum_deaths',
            ],
        })
    elif which == 'overview':
        plots = sc.dcp(overview_plots)
    else: # pragma: no cover
        errormsg = f'The choice which="{which}" is not supported'
        raise ValueError(errormsg)
    return plots

