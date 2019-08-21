library(jsonlite)
library(data.table)
library(RSQLite)

df_air <- data.table()
for (i in c(2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018)){
  df_air_tmp <- data.table(fromJSON(paste0('http://airyards.com/', i, '/weeks')))
  df_air_tmp[, year := i]
  df_air <- rbind(df_air, df_air_tmp)
  
}

setnames(df_air, 'full_name', 'player')

df_air[, index:=NULL]
df_air[, player_id:=NULL]
df_air[, position := NULL]
df_air[, team := NULL]

df_air <- df_air[, lapply(.SD, sum), 
          .SDcols=c('air_yards', 'team_air', 'tm_att', 'tar', 'rec_yards'), 
          by=c('player', 'year')]

df_air[, racr := rec_yards / air_yards]
df_air[, tgt_mkt_share := tar / tm_att]
df_air[, air_yd_mkt_share := air_yards / team_air]
df_air[, wosp := 1.5 * tgt_mkt_share + 0.7 * air_yd_mkt_share]
df_air[, team_air_per_att := team_air / tm_att]
df_air[, air_yd_per_tgt := air_yards / (tar+1)]

df_air[, tm_att := NULL]
df_air[, tar := NULL]
df_air[, rec_yards := NULL]

sqlite <- dbDriver("SQLite")
conn <- dbConnect(sqlite, "/Users/Mark/Documents/Github/Fantasy_Football/Data/Season_Stats.sqlite3")
dbWriteTable(conn, 'AirYards', df_air, overwrite=T)
