"""
Fantasy Football Configuration File
===================================

This file contains all configuration variables used across the Fantasy Football project.
Update these values to change settings across all scripts without editing each file individually.
"""

import datetime as dt

# =============================================================================
# GENERAL SETTINGS
# =============================================================================

# Current year for projections and analysis
YEAR = 2025

# Primary league/version for analysis
LEAGUE = 'beta'  # Options: 'nffc', 'beta', 'dk', 'nv'

# Model version/prediction version
PRED_VERSION = 'final_ensemble'

# Database name for season stats
DB_NAME = 'Season_Stats_New'

# Positions to analyze
POSITIONS = ['QB', 'RB', 'WR', 'TE']

# =============================================================================
# FANTASY SCORING SETTINGS
# =============================================================================

# Rushing scoring by league
RUSH_SCORING = {
    'beta': {
        'rush_yards_gained_sum': 0.1,  
        'rush_rush_touchdown_sum': 7,
        'fumble_lost': -2,
        'rush_yd_100_bonus': 1,
        'rush_yd_200_bonus': 2,
    },
    'dk': {
        'rush_yards_gained_sum': 0.1,
        'rush_rush_touchdown_sum': 6,
        'fumble_lost': -1,
        'rush_yd_100_bonus': 3,
        'rush_yd_200_bonus': 0,
    },
    'nffc': {
        'rush_yards_gained_sum': 0.1,
        'rush_rush_touchdown_sum': 6,
        'fumble_lost': -1,
        'rush_yd_100_bonus': 0,
        'rush_yd_200_bonus': 0,
    },
    'nv': {
        'rush_yards_gained_sum': 0.1,
        'rush_rush_touchdown_sum': 7,
        'fumble_lost': -2,
        'rush_yd_100_bonus': 1,
        'rush_yd_200_bonus': 2,
    }
}

# Receiving scoring by league
RECEIVING_SCORING = {
    'beta': {
        'rec_complete_pass_sum': 0.5, 
        'rec_yards_gained_sum': 0.1,
        'rec_pass_touchdown_sum': 7, 
        'rec_yd_100_bonus': 1,
        'rec_yd_200_bonus': 2,
    },
    'dk': {
        'rec_complete_pass_sum': 1, 
        'rec_yards_gained_sum': 0.1,
        'rec_pass_touchdown_sum': 6, 
        'rec_yd_100_bonus': 3,
        'rec_yd_200_bonus': 0,
    },
    'nffc': {
        'rec_complete_pass_sum': 1,
        'rec_yards_gained_sum': 0.1,
        'rec_pass_touchdown_sum': 6,
        'rec_yd_100_bonus': 0,
        'rec_yd_200_bonus': 0,
    }, 
    'nv': {
        'rec_complete_pass_sum': 0.5,
        'rec_yards_gained_sum': 0.1,
        'rec_pass_touchdown_sum': 7,
        'rec_yd_100_bonus': 1,
        'rec_yd_200_bonus': 2,
    }
}

# Passing scoring by league
PASSING_SCORING = {
    'beta': {
        'pass_yards_gained_sum': 0.04, 
        'pass_pass_touchdown_sum': 5, 
        'pass_interception_sum': -2,
        'sack_sum': -1,
        'pass_yd_300_bonus': 1,
        'pass_yd_400_bonus': 2,
    },
    'dk': {
        'pass_yards_gained_sum': 0.04, 
        'pass_pass_touchdown_sum': 4, 
        'pass_interception_sum': -1,
        'sack_sum': 0,
        'pass_yd_300_bonus': 3,
        'pass_yd_400_bonus': 0,
    },
    'nffc': {
        'pass_yards_gained_sum': 0.05, 
        'pass_pass_touchdown_sum': 6, 
        'pass_interception_sum': -2,
        'sack_sum': 0,
        'pass_yd_300_bonus': 0,
        'pass_yd_400_bonus': 0,
    },
    'nv': {
        'pass_yards_gained_sum': 0.04,
        'pass_pass_touchdown_sum': 4,
        'pass_interception_sum': -2,
        'sack_sum': 2,
        'pass_yd_300_bonus': 1,
        'pass_yd_400_bonus': 2,
    }
}

# Projection scoring (simplified format for calc_proj_pts function)
PROJECTION_SCORING = {
    'pass_yds': PASSING_SCORING[LEAGUE].get('pass_yards_gained_sum', 0.04),
    'pass_td': PASSING_SCORING[LEAGUE].get('pass_pass_touchdown_sum', 6),
    'pass_int': PASSING_SCORING[LEAGUE].get('pass_interception_sum', -2),
    'pass_sacks': PASSING_SCORING[LEAGUE].get('sack_sum', 0),
    'rush_yds': RUSH_SCORING[LEAGUE].get('rush_yards_gained_sum', 0.1),
    'rush_td': RUSH_SCORING[LEAGUE].get('rush_rush_touchdown_sum', 6),
    'rec_yds': RECEIVING_SCORING[LEAGUE].get('rec_yards_gained_sum', 0.1),
    'rec_td': RECEIVING_SCORING[LEAGUE].get('rec_pass_touchdown_sum', 6),
    'rec': RECEIVING_SCORING[LEAGUE].get('rec_complete_pass_sum', 1)
}

# =============================================================================
# DRAFT SETTINGS
# =============================================================================

# Snake draft settings
DRAFT_SETTINGS = {
    'num_teams': 12,
    'num_rounds': 25,
    'replacement_ranks': {'QB': 12, 'RB': 28, 'WR': 44, 'TE': 12}
}

# ADP league for draft analysis
ADP_LEAGUE = LEAGUE

# =============================================================================
# MODEL SETTINGS
# =============================================================================

# Classification cuts for different positions
CLASS_CUTS = {
    'WR': {
        'upside': {'y_act': 24, 'proj_var': 0.7},
        'top': {'y_act': 12, 'proj_var': 0.45}
    },
    'RB': {
        'upside': {'y_act': 24, 'proj_var': 0.7},
        'top': {'y_act': 12, 'proj_var': 0.45}
    },
    'QB': {
        'upside': {'y_act': 12, 'proj_var': 0.7},
        'top': {'y_act': 6, 'proj_var': 0.45}
    },
    'TE': {
        'upside': {'y_act': 12, 'proj_var': 0.7},
        'top': {'y_act': 6, 'proj_var': 0.45}
    },
    'Rookie_WR': {
        'upside': {'y_act': 24, 'proj_var': 0.7},
        'top': {'y_act': 12, 'proj_var': 0.45}
    },
    'Rookie_RB': {
        'upside': {'y_act': 24, 'proj_var': 0.7},
        'top': {'y_act': 12, 'proj_var': 0.45}
    },
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_scoring_dict(scoring_type, league=None):
    """
    Get scoring dictionary for a specific type and league.
    
    Parameters:
    -----------
    scoring_type : str
        Type of scoring ('rush', 'receiving', 'passing', 'projection')
    league : str, optional
        League name. Defaults to LEAGUE constant.
    
    Returns:
    --------
    dict : Scoring values for the specified type and league
    """
    if league is None:
        league = LEAGUE
    
    if scoring_type == 'rush':
        return RUSH_SCORING.get(league, RUSH_SCORING['nffc'])
    elif scoring_type == 'receiving':
        return RECEIVING_SCORING.get(league, RECEIVING_SCORING['nffc'])
    elif scoring_type == 'passing':
        return PASSING_SCORING.get(league, PASSING_SCORING['nffc'])
    elif scoring_type == 'projection':
        return PROJECTION_SCORING
    else:
        raise ValueError(f"Unknown scoring type: {scoring_type}")

def get_max_pick():
    """Calculate maximum pick number for draft analysis"""
    return DRAFT_SETTINGS['num_teams'] * DRAFT_SETTINGS['num_rounds']

# =============================================================================
# VALIDATION
# =============================================================================

def validate_config():
    """Validate configuration settings"""
    valid_leagues = list(RUSH_SCORING.keys())
    if LEAGUE not in valid_leagues:
        raise ValueError(f"LEAGUE must be one of {valid_leagues}")
    
    if YEAR < 2020 or YEAR > 2030:
        print(f"Warning: YEAR ({YEAR}) seems unusual. Please verify.")
    
    return True

# Run validation on import
validate_config()
