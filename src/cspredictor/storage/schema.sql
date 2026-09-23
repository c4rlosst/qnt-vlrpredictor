PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hltv_id INTEGER UNIQUE,
    name TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_teams_name ON teams(name);

CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hltv_id INTEGER UNIQUE,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hltv_id INTEGER UNIQUE,
    name TEXT,
    is_lan INTEGER,           -- 0/1/NULL (unknown)
    tier TEXT
);

CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hltv_id INTEGER UNIQUE NOT NULL,
    match_url TEXT,
    event_id INTEGER REFERENCES events(id),
    match_date TEXT,          -- ISO 8601 date, derived from unix_timestamp_ms when available
    unix_timestamp_ms INTEGER,
    team1_id INTEGER REFERENCES teams(id),
    team2_id INTEGER REFERENCES teams(id),
    best_of INTEGER,
    team1_score INTEGER,      -- maps won
    team2_score INTEGER,
    winner_team_id INTEGER REFERENCES teams(id),
    is_lan INTEGER,
    scraped_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date);
CREATE INDEX IF NOT EXISTS idx_matches_team1 ON matches(team1_id);
CREATE INDEX IF NOT EXISTS idx_matches_team2 ON matches(team2_id);

CREATE TABLE IF NOT EXISTS maps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL REFERENCES matches(id),
    map_order INTEGER NOT NULL,
    map_name TEXT,
    team1_score INTEGER,
    team2_score INTEGER,
    winner_team_id INTEGER REFERENCES teams(id),
    picked_by_team_id INTEGER REFERENCES teams(id),  -- NULL => decider/leftover
    UNIQUE(match_id, map_order)
);
CREATE INDEX IF NOT EXISTS idx_maps_match ON maps(match_id);

CREATE TABLE IF NOT EXISTS rosters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL REFERENCES matches(id),
    team_id INTEGER NOT NULL REFERENCES teams(id),
    player_id INTEGER NOT NULL REFERENCES players(id),
    UNIQUE(match_id, team_id, player_id)
);
CREATE INDEX IF NOT EXISTS idx_rosters_match ON rosters(match_id);
CREATE INDEX IF NOT EXISTS idx_rosters_team ON rosters(team_id);

CREATE TABLE IF NOT EXISTS ranking_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER NOT NULL REFERENCES teams(id),
    snapshot_date TEXT NOT NULL,  -- ISO 8601 date
    rank INTEGER,
    points INTEGER,
    UNIQUE(team_id, snapshot_date)
);
CREATE INDEX IF NOT EXISTS idx_rankings_date ON ranking_snapshots(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_rankings_team ON ranking_snapshots(team_id);

-- key/value store used by the backfill script to resume an interrupted run
CREATE TABLE IF NOT EXISTS scrape_state (
    key TEXT PRIMARY KEY,
    value TEXT
);
