lr_mean, lr_std, test_template = expected_pts_model(d)

my_team_select = pd.DataFrame({'Position':['RB', 'RB', 'WR', 'QB'], 
                                'Player': ['Alvin Kamara', 'Nick Chubb', 'DJ Moore', 'Sam Darnold'], 
                                'Salary':[55, 33, 33, 1]})
my_team_update = my_team_df.copy()

def expected_pts(team_template, my_team_select, test_template):
    '''
    INPUT: Blank team template dataframe, dataframe of selected players, and the X_test
           template the auto formats the one-hot-encoding for points prediction
    
    OUTPUT: A list with expected points added to your team based on average points 
            scored by the player vs what would be expected based on position + salary regression
    '''

    #----------
    # Create the initial baseline projection for your team
    #----------

    # copy the blank team template
    my_team_template = team_template.copy()

    # determine the remaining salary for the team across positions
    remain_base, _ = salary_proportions(my_team_template, proport, league_info['salary_cap'])
    
    # create the X prediction for the base case and determine point distribution
    X_base = create_X_test(remain_base, test_template)
    team_dist_base = create_pt_dist(X_base, selected, lr_mean, lr_std)

    # loop through each player in selected players
    exp_pts = []
    for _, row in my_team_select.iterrows():

        # get the team dataframe filled with only the current player
        my_team_template = team_template.copy()
        my_team_update = team_fill(my_team_template, pd.DataFrame(row).T)
        remain_pl, selected_pl = salary_proportions(my_team_update, proport, 
                                                    league_info['salary_cap']-row.Salary)
        
        # predict points scored by the current player and all other blanks
        X_pl = create_X_test(remain_pl, test_template)
        team_dist_pl = create_pt_dist(X_pl, selected_pl, lr_mean, lr_std)
        
        # determine the number of points added by this player
        pts_added = (team_dist_pl.mean() - team_dist_base.mean()).values[0]
        exp_pts.append([row.Player, round(pts_added, 2)])

    return pd.DataFrame(exp_pts, columns=['Player', 'AddedPoints'])

expected_pts(my_team_update, my_team_select, test_template)