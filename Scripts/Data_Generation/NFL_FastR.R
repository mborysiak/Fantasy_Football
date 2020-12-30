library(nflfastR)
library(arrow)
library(data.table)

#------------------
# Pull in the Play-by-Play Data
#------------------

# define which seasons shall be loaded
seasons <- 1999:2020
pbp <- purrr::map_df(seasons, function(x) {
  readRDS(
    url(glue::glue("https://raw.githubusercontent.com/guga31bb/nflfastR-data/master/data/play_by_play_{x}.rds")))
})

# convert the player id's and pass into datatable
pbp <- nflfastR::decode_player_ids(pbp)
pbp <- data.table(pbp)

# remove full player names and filter down to real players
pbp[, c('receiver_player_name', 'rusher_player_name', 'passer_player_name') := NULL]
pbp <- pbp[player==1]

#------------------
# Pull in the Roster Data
#------------------

# path to the player roster data + read it in
ROSTER_URL <- 'https://raw.githubusercontent.com/guga31bb/nflfastR-data/master/roster-data/roster.rds'
rosters <- data.table(readRDS(url(ROSTER_URL)))

NFL_FASTR_ROSTER <- 'https://raw.githubusercontent.com/mrcaseb/nflfastR-roster/master/data/nflfastR-roster.rds'
new_rosters <- data.table(readRDS(NFL_FASTR_ROSTER)))

rosters <- purrr::map_df(seasons, function(x) {
  readRDS(url(glue::glue('https://raw.githubusercontent.com/mrcaseb/nflfastR-roster/master/data/seasons/roster_{x}.rds')))
})
rosters <- data.table(rosters)


# select relevant columns from dataset
old_cols <- c('team.season', 'teamPlayers.gsisId', 'teamPlayers.displayName',
              'teamPlayers.positionGroup',  'teamPlayers.position',
              'teamPlayers.birthDate',  'teamPlayers.collegeName', 'teamPlayers.height',
              'teamPlayers.weight')

new_cols <- c('season', 'player_id', 'player_name', 'player_position_group',
              'player_position', 'player_birthdate', 'player_college_name',
              'player_height', 'player_weight')

# select relevant columns from dataset and rename the columns to better names
rosters <- rosters[, ..old_cols]
rosters <- setnames(rosters, old_cols, new_cols)
rosters <- rosters[!is.na(player_id)]

#------------------
# Merge the Play-by-Play and Roster Data
#------------------

merge_stats <- function(player_type){
  
  player_type_cols <- c('season')
  for (i in colnames(rosters)[2:length(new_cols)]){
    i = paste0(player_type, '_',i)
    player_type_cols <- c(player_type_cols, i)
    
  }
  
  rosters <- setnames(rosters, new_cols, player_type_cols)
  
  pbp <- merge.data.table(pbp, rosters, all.x=TRUE, by=c('season', paste0(player_type, '_', 'player_id')))
  rosters <- setnames(rosters, player_type_cols, new_cols)
  
  return(pbp)
}
pbp <- merge_stats('receiver')
pbp <- merge_stats('rusher')
pbp <- merge_stats('passer')

# calculate ages based on season and players birthday
pbp[, receiver_player_age:=as.numeric((as.Date(paste0(season, '-09-01'))-as.Date(receiver_player_birthdate, format='%m/%d/%Y')) / 365)]
pbp[, rusher_player_age:=as.numeric((as.Date(paste0(season, '-09-01'))-as.Date(rusher_player_birthdate, format='%m/%d/%Y')) / 365)]
pbp[, passer_player_age:=as.numeric((as.Date(paste0(season, '-09-01'))-as.Date(passer_player_birthdate, format='%m/%d/%Y')) / 365)]

# filter down the dataset to only rushing and receiving plays
pbp <- pbp[play_type %in% c('run', 'pass')]

# save out parquets of the data
arrow::write_parquet(pbp, '/Users/mborysia/Documents/Github/Fantasy_Football/Data/OtherData/NFL_FastR/raw_data20201224.parquet')
arrow::write_parquet(rosters, '/Users/mborysia/Documents/Github/Fantasy_Football/Data/OtherData/NFL_FastR/rosters20201224.parquet')