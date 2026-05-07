import pathlib
from urllib import response
import xml.etree.ElementTree as ET
import pickle
import bs4
import multiprocessing
import sqlite3
import datetime
import pandas as pd
import data_download

months = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
          'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

# Database schema, persistance, and queries:
def open_sqlite3_chess_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row      # Allows us to access columns by name.

    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # Check if tables YearMonthElos and YearMonthParseCompletions exist, if not create them.
    if ('YearMonthElos',) not in tables:
        cursor.execute("CREATE TABLE IF NOT EXISTS YearMonthElos (iid INTEGER PRIMARY KEY, year INTEGER, month INTEGER, fideid INTEGER, name TEXT, country TEXT, sex TEXT, title TEXT, w_title TEXT, o_title TEXT, rating INTEGER, games INTEGER, k INTEGER, birthday TEXT, flag TEXT)")
    if ('YearMonthParseCompletions',) not in tables:
        cursor.execute("CREATE TABLE IF NOT EXISTS YearMonthParseCompletions (year INTEGER, month INTEGER, PRIMARY KEY (year, month))")
    if ('PlayerMaxElosByYearMonth',) not in tables:
        cursor.execute("CREATE TABLE IF NOT EXISTS PlayerMaxElosByYearMonth (fideid INTEGER, year INTEGER, month INTEGER, max_elo INTEGER, PRIMARY KEY (fideid, year, month))")
    conn.commit()

    return conn, cursor

def commit_year_month_elos_to_db(cursor: sqlite3.Cursor, year_month, players):
    for player in players.values():
        cursor.execute("INSERT INTO YearMonthElos (year, month, fideid, name, country, sex, title, w_title, o_title, rating, games, k, birthday, flag) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                          (year_month.year, year_month.month, player.fideid, player.name, player.country, player.sex, player.title, player.w_title, player.o_title, player.rating, player.games, player.k, player.birthday, player.flag))
    cursor.execute("INSERT INTO YearMonthParseCompletions (year, month) VALUES (?, ?)", (year_month.year, year_month.month))
    cursor.connection.commit()

def commit_year_month_elos_to_db_if_not_exists(cursor: sqlite3.Cursor, year_month, players):
    if check_year_month_parsed(cursor, year_month):
        print(f"Data for {year_month} already parsed, skipping.")
        return
    commit_year_month_elos_to_db(cursor, year_month, players)

def check_year_month_parsed(cursor: sqlite3.Cursor, year_month):
    cursor.execute("SELECT 1 FROM YearMonthParseCompletions WHERE year = ? AND month = ?", (year_month.year, year_month.month))
    return cursor.fetchone() is not None

def get_elos_for_year_month_from_db(cursor: sqlite3.Cursor, year_month):
    cursor.execute("SELECT fideid, name, country, sex, title, w_title, o_title, rating, games, k, birthday, flag FROM YearMonthElos WHERE year = ? AND month = ?", (year_month.year, year_month.month))
    rows = cursor.fetchall()
    players = {}
    for row in rows:
        fideid, name, country, sex, title, w_title, o_title, rating, games, k, birthday, flag = row
        players[fideid] = PlayerMonth(fideid, name, country, sex, title, w_title, o_title, rating, games, k, birthday, flag)
    return players

def persist_full_elo_history_to_db(all_data, cursor: sqlite3.Cursor):
    for year_month, players in all_data.items():
        commit_year_month_elos_to_db_if_not_exists(cursor, year_month, players)

def get_max_elo_for_all_players(cursor: sqlite3.Cursor, cutoff_year_month):
    # First we check if the PlayerMaxElos table is populated, if so we can just return that. Otherwise we compute it from the YearMonthElos table and populate the PlayerMaxElos table for future use.
    cursor.execute("SELECT fideid, max_elo FROM PlayerMaxElosByYearMonth WHERE year = ? AND month = ?", (cutoff_year_month.year, cutoff_year_month.month))
    rows = cursor.fetchall()
    if len(rows) > 0:
        return {row[0]: row[1] for row in rows}
    
    cursor.execute("""INSERT INTO PlayerMaxElosByYearMonth (fideid, year, month, max_elo)
                   SELECT fideid, ?, ?, MAX(rating)
                   FROM YearMonthElos
                   WHERE (year < ? OR (year = ? AND month <= ?))
                   GROUP BY fideid""", (cutoff_year_month.year, cutoff_year_month.month, cutoff_year_month.year, cutoff_year_month.year, cutoff_year_month.month))
    cursor.connection.commit()
    
    return get_max_elo_for_all_players(cursor, cutoff_year_month)

def get_all_data_for_players(cursor: sqlite3.Cursor, fideids):
    cursor.execute("SELECT year, month, fideid, name, country, sex, title, w_title, o_title, rating, games, k, birthday, flag FROM YearMonthElos WHERE fideid IN ({seq})".format(seq=','.join(['?']*len(fideids))), fideids)
    rows = cursor.fetchall()
    
    player_data = {}    # player_data[fideid][year_month] = playermonth data row.

    for row in rows:
        fideid = row["fideid"]
        year_month = YearMonth(row["year"], row["month"])
        if fideid not in player_data:
            player_data[fideid] = {}
        player_data[fideid][year_month] = row

    return player_data

# Parsing and committing
class PlayerMonth:
    """Convenience class for temporarily storing parsed player month data. This is not meant to be used outside of the parsing functions."""
    __slots__ = ['fideid', 'name', 'country', 'sex', 'title', 'w_title', 'o_title', 'rating', 'games', 'k', 'birthday', 'flag']

    def __init__(self, fideid, name, country, sex, title, w_title, o_title, rating, games, k, birthday, flag):
        self.fideid = fideid
        self.name = name
        self.country = country
        self.sex = sex
        self.title = title
        self.w_title = w_title
        self.o_title = o_title
        self.rating = rating
        self.games = games
        self.k = k
        self.birthday = birthday
        self.flag = flag
    
    def __repr__(self) -> str:
        return f"PlayerMonth(fideid={self.fideid}, name='{self.name}')"

class YearMonth:
    __slots__ = ['year', 'month']

    def __init__(self, year, month):
        self.year = year
        self.month = month
    
    @staticmethod
    def parse_from_new_fide_filename(file_name: pathlib.Path):
        base_name = file_name.stem  # Remove .zip or .xml
        parts = base_name.split('_')
        if len(parts) < 3:
            raise ValueError(f"Unexpected file name format: {file_name}")
        
        month_year_part = parts[1]  # e.g., 'jan15'
        month_str = month_year_part[:3]
        year_str = month_year_part[3:5]

        if month_str not in months:
            raise ValueError(f"Unknown month in file name: {month_str}")
        
        month = months[month_str]
        year = int('20' + year_str)  # Convert '15' to 2015

        return YearMonth(year, month)
    
    @staticmethod
    def parse_from_old_fide_filename(file_name: pathlib.Path):
        base_name = file_name.stem  # Remove .zip or .TXT
        
        month_year_str = base_name[0:5].lower()  # e.g., 'jan15'
        month_str = month_year_str[:3]
        year_str = month_year_str[3:5]

        if month_str not in months:
            raise ValueError(f"Unknown month in file name: {month_str}")
        
        month = months[month_str]
        year = int('20' + year_str)  # Convert '15' to 2015

        return YearMonth(year, month)

    @staticmethod
    def parse_from_olimpbase_filename(file_name: pathlib.Path):
        base_name = file_name.stem  # Remove .zip or .html
        
        month_year_str = base_name[3:10].lower()  # e.g., 'jan15'
        month_str = month_year_str[:3]
        year_str = month_year_str[3:]

        if month_str not in months:
            raise ValueError(f"Unknown month in file name: {month_str}")
        
        month = months[month_str]
        year = int(year_str)

        return YearMonth(year, month)
    
    def short_form(self) -> str:
        month_str = [k for k, v in months.items() if v == self.month][0]
        year_str = str(self.year)[2:]
        return f"{month_str}{year_str}"
    
    def year_month_to_datetime(self):
        return datetime.datetime(self.year, self.month, 1)  # Elo lists assumed to be released at the start of each month.
    
    def __repr__(self) -> str:
        return f"YearMonth(year={self.year}, month={self.month})"
    
    def __hash__(self) -> int:
        return hash((self.year, self.month))

    def __lt__(self, other):
        if self.year == other.year:
            return self.month < other.month
        return self.year < other.year
    
    def __le__(self, other):
        return self < other or self.year == other.year and self.month == other.month

def parse_new_format_monthly_data(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    players = {}
    for player in root.findall('player'):
        fideid = player.find('fideid').text
        name = player.find('name').text
        country = player.find('country').text
        sex = player.find('sex').text
        title = player.find('title').text
        w_title = player.find('w_title').text
        o_title = player.find('o_title').text
        rating = int(player.find('rating').text)
        games = int(player.find('games').text)
        k = int(player.find('k').text)

        try:
            birthday = f"01/01/{player.find('birthday').text}"
        except (TypeError, ValueError):
            birthday = None

        flag = player.find('flag').text

        players[fideid] = PlayerMonth(fideid, name, country, sex, title, w_title, o_title, rating, games, k, birthday, flag)

    print("Loaded data for: ", file_path)
    return players

def special_old_format_cases(line):
    broken_lines = {
        " 6500889                                        CRC  1969    0        i\n": " 6500889   Mubayiwa, Bruce             CRC  1969    0        .",
    }

    if line in broken_lines:
        return broken_lines[line]
    return line

def parse_old_format_monthly_data(file_path, year_month: YearMonth):
    print(f"Loading data for: {file_path}")

    # The old FIDE data format is very odd, we have to rely on the spacing of the header line to parse the data.
    ORIGINAL_VERSION_FIELDS = ["ID_NUMBER", "NAME", "TITLE", "COUNTRY", year_month.short_form().upper(), "GAMES", "BIRTHDAY", "FLAG"]
    SECOND_VERSION_FIELDS = ["ID number", "Name", "Titl", "Fed", year_month.short_form().capitalize(), "Games", "Born", "Flag"]

    with open(file_path, 'r', encoding='ISO-8859-1') as f:
        lines = f.readlines()
    header_line = lines[0]
    original_fields_starts = [header_line.find(field) for field in ORIGINAL_VERSION_FIELDS]
    second_fields_starts = [header_line.find(field) for field in SECOND_VERSION_FIELDS]

    players = {}
    start_line = 1

    if sum(1 for start in original_fields_starts if start != -1) >= 7 or sum(1 for start in second_fields_starts if start != -1) >= 7:      # Some have one field missing or a mistake in the month/year
        pass
    elif year_month.year == 2003 and year_month.month == 10 or \
         year_month.year == 2004 and year_month.month == 1 or \
         year_month.year == 2003 and year_month.month == 7 or \
         year_month.year == 2005 and year_month.month == 4:
        start_line = 0  # Manually verified that the header is missing from this file
    else:
        raise ValueError(f"Unexpected header format in file: {file_path}")

    # January 2001 doesn't have birthdays, so we have to use a special parsing method for that month.
    # We also use this method for some finnicky years.
    if year_month.year < 2002 or year_month.year == 2002 and year_month.month < 10:
        for line in lines[start_line:]:
            if line.strip() == "":
                continue
            
            player_id = int(line[original_fields_starts[0]:original_fields_starts[1]].strip())
            name = line[original_fields_starts[1]:original_fields_starts[2]].strip()
            title = line[original_fields_starts[2]:original_fields_starts[3]].strip()
            country = line[original_fields_starts[3]:original_fields_starts[4]].strip()
            elo = int(line[original_fields_starts[4]:original_fields_starts[5]].strip())

            if original_fields_starts[6] != -1:
                games = int(line[original_fields_starts[5]:original_fields_starts[6]].strip())
                birthday = line[original_fields_starts[6]:original_fields_starts[7]].strip().replace('.', '/')
            else:
                games = int(line[original_fields_starts[5]:original_fields_starts[7]].strip())
                birthday = None
            flag = line[original_fields_starts[7]:].strip()

            if 'w' in flag: # Womens competitor. Weird format pre 2012.
                womens_title = title
                sex = 'F'

                if "(GM)" in name:
                    title = "g"
                elif "(IM)" in name:
                    title = "m"
            else:
                womens_title = ""
                sex = 'M'

            players[player_id] = PlayerMonth(player_id, name, country, sex, title, womens_title, None, elo, games, None, birthday, flag)
    else:
        # Need to completely redo this based on number of spaces between fields...
        for i, line in enumerate(lines[start_line:]):
            line = special_old_format_cases(line)
            if (line := line.strip()) == "":
                continue
            if i == len(lines[start_line:]) - 1 and len(line) < 10:
                continue
            
            # First a number (ID), then a name (which may contain up to 1 consecutive space), then a title (which may be empty, but if not is 1-2 characters), then a mandatory 3 letter country code, then ELO, then number of games played, year of birth (optional), and finally an optional flag (non-numeric).
            # We can't rely on secondfield starts, so we'll just have to parse based on the spacing of individual lines.
            cur_char = 0
            id_str = ""
            while line[cur_char] in '1234567890': # ID number
                id_str += line[cur_char]
                cur_char += 1
            player_id = int(id_str)

            name_str = ""
            spaces_in_a_row = 0
            last_non_space_was_comma = False
            seen_non_space = False

            while not seen_non_space or ((len(name_str) < 30 and spaces_in_a_row < 5) or spaces_in_a_row < 2 or last_non_space_was_comma and spaces_in_a_row < 5):    # Some mid-name spacing issues...
                if line[cur_char] == ' ':
                    spaces_in_a_row += 1
                else:
                    spaces_in_a_row = 0
                    seen_non_space = True
                    if line[cur_char] == ',':
                        last_non_space_was_comma = True
                    else:
                        last_non_space_was_comma = False
                name_str += line[cur_char]
                cur_char += 1
            name = name_str.strip()

            remainder_parts = line[cur_char:].strip().split()
            if len(remainder_parts[0]) <= 2:
                title = remainder_parts[0]
                remainder_parts = remainder_parts[1:]
            else:
                title = ""
            
            country = remainder_parts[0]
            elo = int(remainder_parts[1])
            games = int(remainder_parts[2])

            if len(remainder_parts) > 3:
                if len(remainder_parts[3]) == 4 or len(remainder_parts[3]) == 8: # Birthday is present
                    birthday = remainder_parts[3]
                    if len(birthday) == 4:
                        birthday = f"01/01/{birthday}"
                    else:
                        birthday = birthday.replace('.', '/')
                    flag = remainder_parts[4] if len(remainder_parts) > 4 else ""
                elif remainder_parts[3] == '.': # Some special formatting for missing birthdays on some txt files.
                    birthday = None
                    flag = remainder_parts[5] if len(remainder_parts) > 5 else ""
                else:
                    birthday = None
                    flag = remainder_parts[3] if len(remainder_parts) > 3 else ""
            else:
                birthday = None
                flag = ""

            if 'w' in flag: # Womens competitor. Weird format pre 2012.
                womens_title = title
                sex = 'F'

                if "(GM)" in name:
                    title = "g"
                elif "(IM)" in name:
                    title = "m"
            else:
                womens_title = ""
                sex = 'M'

            players[player_id] = PlayerMonth(player_id, name, country, sex, title, womens_title, None, elo, games, None, birthday, flag)
    
    print("Loaded data for: ", year_month)
    return players

def commit_all_old_format_monthly_data(standard_data_path, cursor: sqlite3.Cursor):
    all_files = list(standard_data_path.glob('*.TXT')) + list(standard_data_path.glob('*.txt'))

    for file in all_files:
        year_month = YearMonth.parse_from_old_fide_filename(file)
        if check_year_month_parsed(cursor, year_month):
            print(f"Data for {year_month} already parsed, skipping.")
            continue
        
        players = parse_old_format_monthly_data(file, year_month)
        commit_year_month_elos_to_db_if_not_exists(cursor, year_month, players)

def int_or_none(s:str):
    try:
        return int(s)
    except (ValueError, TypeError):
        return None

def nonempty_str_or_none(s:str):
    if s == "":
        return None
    return s

def parse_olimpbase_monthly_data(file_path):
    # The olimp data is formatted as a <pre> html table. So we parse it based on the *visible* spacing of the table header.
    # Note, the header is the same for every month, but some fields only start being populated later on, such as FIDE ID. These will have to be reconcilliated later.

    FIELD_NAMES = ["pos", "Player_ID", "Name", "Title", "Fed", "Rtng", "+/-", "gms", "Birthday", "Sex", "Flag"]
    header_substring = "pos Player_ID  Name                                  Title Fed  Rtng   +/-  gms   Birthday   Sex  Flag"
    
    try:
        with open(file_path, 'r') as f:
            soup = bs4.BeautifulSoup(f, 'html.parser')
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='ISO-8859-1') as f:
            soup = bs4.BeautifulSoup(f, 'html.parser')
    
    players = [] # FIDE IDs don't neccessarily exist yet, need to do reconcilliation.

    text = soup.text
    header_yet = False
    header_positions = []

    for line in text.splitlines():
        if not header_yet:
            if header_substring in line:
                header_yet = True
                for field in FIELD_NAMES:
                    header_positions.append(line.find(field))
            continue

        if line.strip() == "":
            continue
        elif "Inactive players" in line:
            continue
        elif header_substring in line:
            continue

        # Parsing the actual data.
        player_id = int_or_none(line[header_positions[1]:header_positions[2]].strip())
        name = line[header_positions[2]:header_positions[3]].strip()
        title = nonempty_str_or_none(line[header_positions[3]:header_positions[4]].strip())
        country = nonempty_str_or_none(line[header_positions[4]:header_positions[5]].strip())
        rating = int_or_none(line[header_positions[5]:header_positions[6]].strip())
        games = int_or_none(line[header_positions[7]:header_positions[8]].strip())
        birthday = nonempty_str_or_none(line[header_positions[8]:header_positions[9]].strip())
        flag = line[header_positions[10]:].strip()

        if name == "":
            continue # TODO: Remove breakpoints

        if 'w' in flag: # Womens competitor. Weird format pre 2012.
            womens_title = title
            sex = 'F'

            if "(GM)" in name:
                title = "g"
            elif "(IM)" in name:
                title = "m"
        else:
            womens_title = ""
            sex = 'M'
        
        players.append(PlayerMonth(player_id, name, country, sex, title, womens_title, None, rating, games, None, birthday, flag))
    return players

def commit_all_olimpbase_monthly_data(cursor: sqlite3.Cursor):
    all_data = {}

    files = list(pathlib.Path('data/standard/olimpbase').glob('*.html'))
    year_months = [YearMonth.parse_from_olimpbase_filename(file) for file in files]

    if all(check_year_month_parsed(cursor, ym) for ym in year_months):
        print("Olimpbase data already parsed, skipping.")
        return

    for file in files:
        year_month = YearMonth.parse_from_olimpbase_filename(file)  # Yes, this is slightly redundant.
        players = parse_olimpbase_monthly_data(file)
        all_data[year_month] = players
        print("Loaded data for: ", year_month)
    
    # ID Reconcilliation.
    print("Starting ID reconcilliation...")
    def normalize_player_name(name):
        return name.lower().replace(' ', '').replace('-', '').replace('.', '').replace(",", "")
    
    normalized_player_names_to_id = {}
    unreconcilable_players = set()
    fake_id = -1
    
    for ym in all_data.__reversed__():
        for player in all_data[ym]:
            normalized_name = normalize_player_name(player.name)
            if normalized_name in normalized_player_names_to_id:
                player.fideid = normalized_player_names_to_id[normalized_name]
            elif player.fideid is not None:
                normalized_player_names_to_id[normalized_name] = player.fideid
            else:
                unreconcilable_players.add(player.name)
                normalized_player_names_to_id[normalized_name] = fake_id
                player.fideid = fake_id
                fake_id -= 1
                
    print(f"ID reconcilliation complete. Num unreconcilable players: {len(unreconcilable_players)}")
    print("Unreconcilable players will be given a fake ID between -1 and -infty")

    all_data = {ym: {player.fideid: player for player in players} for ym, players in all_data.items()}
    persist_full_elo_history_to_db(all_data, cursor)
    
def commit_all_standard_monthly_data(cursor: sqlite3.Cursor):
    standard_data_path = pathlib.Path('data/standard')
    files = list(standard_data_path.glob('*.xml'))

    for file in files:
        year_month = YearMonth.parse_from_new_fide_filename(file)

        if check_year_month_parsed(cursor, year_month):
            print(f"Data for {year_month} already parsed, skipping.")
            continue
        players = parse_new_format_monthly_data(file)
        commit_year_month_elos_to_db(cursor, year_month, players)

def high_elo_player_data(cursor: sqlite3.Cursor, cutoff_year_month, elo_threshold=2500):
    max_elos = get_max_elo_for_all_players(cursor, cutoff_year_month)
    high_elo_players = {fideid: max_elo for fideid, max_elo in max_elos.items() if max_elo >= elo_threshold}

    return get_all_data_for_players(cursor, list(high_elo_players.keys()))

def parse_and_save_standard_data(sqlite_db_path):
    conn, cursor = open_sqlite3_chess_db(sqlite_db_path)

    commit_all_olimpbase_monthly_data(cursor)
    commit_all_old_format_monthly_data(pathlib.Path('data/standard'), cursor)
    commit_all_standard_monthly_data(cursor)

    return None

def open_filtered_standard_data(sqlite_db_path = pathlib.Path("data/standard/chess_elos.db"), elo_threshold=2500, cutoff_year_month=YearMonth(2024, 1)):  # Reference study cuts off at January 2024.
    data_download.download_and_unpack_fide_standard_data()

    print("Parsing and saving standard data to database...")
    parse_and_save_standard_data(sqlite_db_path)
    print("Parsing complete, loading data from database...")

    conn, cursor = open_sqlite3_chess_db(sqlite_db_path)
    high_elo_data = high_elo_player_data(cursor, cutoff_year_month, elo_threshold)
    return high_elo_data

class PlayerEloMetadata:
    ELO_CUTOFFS = [2000, 2400, 2500, 2600, 2700, 2800]
    
    MAX_ELO = 0
    MAX_ELO_DATE = YearMonth(1900, 1)

    def __init__(self, player_months: dict[YearMonth, object]):
        self.first_elo_dates = {cutoff: None for cutoff in self.ELO_CUTOFFS}
        abs_elo_changes = {cutoff: [] for cutoff in self.ELO_CUTOFFS}
        self.first_gm_date = None

        games_played = []
        last_elo = 0
        last_ym = YearMonth(1900, 1)
        for ym in sorted(player_months.keys()):
            row = player_months[ym]

            already_passed_2000 = self.first_elo_dates[2000] is not None    # We need to track this for later. Since we don't want to divide by zero.

            if row["rating"] is None:
                continue

            for elo in self.ELO_CUTOFFS:
                if row["rating"] >= elo and (self.first_elo_dates[elo] is None or ym < self.first_elo_dates[elo]):
                    self.first_elo_dates[elo] = ym
                if row["title"] is not None and 'g' in row["title"].lower() and 'w' not in row['title'].lower() and (self.first_gm_date is None or ym < self.first_gm_date):
                    self.first_gm_date = ym
            
            if row["rating"] > self.MAX_ELO:
                self.MAX_ELO = row["rating"]
                self.MAX_ELO_DATE = ym
            
            if already_passed_2000:
                num_months = (ym.year - last_ym.year) * 12 + (ym.month - last_ym.month)
                for elo in self.ELO_CUTOFFS:
                    if self.first_elo_dates[elo] is None:
                        abs_elo_changes[elo].append(abs(row["rating"] - last_elo)/num_months)
                
                if row["games"] is not None:    # Early data doesn't include # games played.
                    games_played.append(row["games"]/num_months)
            
            last_elo = row["rating"]
            last_ym = ym
        
        self.avg_abs_elo_changes_per_month = {
            elo: sum(changes)/len(changes) if len(changes) > 0 and self.first_elo_dates[elo] is not None else None
            for elo, changes in abs_elo_changes.items()
        }
        self.avg_games_per_month_since_2000 = sum(games_played)/len(games_played) if len(games_played) > 0 else None

    def __repr__(self) -> str:
        return f"PlayerEloMetadata(first_elo_dates={self.first_elo_dates}, first_gm_date={self.first_gm_date})"

def get_player_personal_information(player_months: dict[YearMonth, object]):
    # We take this from the latest month we have data for.
    latest_month = max(player_months.keys())
    row = player_months[latest_month]
    return PlayerPersonalInformation(row["fideid"], row["name"], row["country"], row["sex"], row["birthday"])

def birthday_string_to_datetime(birthday_str):
    try:
        if len(birthday_str) == 4:  # Year only
            return datetime.datetime(int(birthday_str), 6, 1) # Assume June 1st for players with only year of birth
        elif '.' in birthday_str: #Year.Month.Day format
            if len(birthday_str) == 9: # Missing leading 1.
                birthday_str = '1' + birthday_str
            return datetime.datetime.strptime(birthday_str, "%Y.%m.%d")
        elif '/' in birthday_str: # Day/Month/Year format
            return datetime.datetime.strptime(birthday_str, "%d/%m/%Y")
    except Exception as e:
        pass

    return datetime.datetime(1900, 1, 1) # Default value for invalid or missing birthdays.

class TabularModelData:
    def __init__(self, model_name, data_table, X_columns, Y_columns):
        self.model_name = model_name
        self.data_table = data_table
        self.X_columns = X_columns
        self.Y_columns = Y_columns
    
    def __repr__(self) -> str:
        return f"TabularModelData(model_name='{self.model_name}', n={len(self.data_table)})"

def build_tables_for_paper_models(player_data, return_condition_sets_and_personal_info = False):
    print("Building tables for paper models...")
    base_table = pd.DataFrame(columns=[
        "fideid",
        "sex",
        "birthday",
        "elo2400_date",
        "elo2500_date",
        "elo2600_date",
        "elo2700_date",
        "elo2400_to_2500_days",
        "elo2500_to_2600_days",
        "elo2600_to_2700_days",
        "max_elo",
        "max_elo_date",
        "gm_title_date",
        "max_elo_age",
        "gm_title_age",

        # Experimental new features.
        # 2000 chosen as a somewhat arbitrary milestone, excluding early play.
        "2000_2500_avg_abs_elo_change_per_month",
        "2000_2600_avg_abs_elo_change_per_month",
        "avg_games_per_month_since_2000"])

    personal_infos = {}
    elo_first_dates_dict = {}

    for fideid, player_months in player_data.items():
        personal_info = get_player_personal_information(player_months)

        if return_condition_sets_and_personal_info:
            personal_infos[fideid] = personal_info

        elo_first_dates = PlayerEloMetadata(player_months)
        if return_condition_sets_and_personal_info:
            elo_first_dates_dict[fideid] = elo_first_dates

        birthday_datetime = personal_info.birthday_datetime

        base_table.loc[len(base_table)] = {
            "fideid": fideid,
            "sex": personal_info.sex,
            "birthday": birthday_datetime if personal_info.birthday is not None else pd.NaT,
            "elo2400_date": elo_first_dates.first_elo_dates[2400].year_month_to_datetime() if elo_first_dates.first_elo_dates[2400] is not None else pd.NaT,
            "elo2500_date": elo_first_dates.first_elo_dates[2500].year_month_to_datetime() if elo_first_dates.first_elo_dates[2500] is not None else pd.NaT,
            "elo2600_date": elo_first_dates.first_elo_dates[2600].year_month_to_datetime() if elo_first_dates.first_elo_dates[2600] is not None else pd.NaT,
            "elo2700_date": elo_first_dates.first_elo_dates[2700].year_month_to_datetime() if elo_first_dates.first_elo_dates[2700] is not None else pd.NaT,
            "elo2400_to_2500_days": (elo_first_dates.first_elo_dates[2500].year_month_to_datetime() - elo_first_dates.first_elo_dates[2400].year_month_to_datetime()).days if elo_first_dates.first_elo_dates[2500] is not None else float('nan'),
            "elo2500_to_2600_days": (elo_first_dates.first_elo_dates[2600].year_month_to_datetime() - elo_first_dates.first_elo_dates[2500].year_month_to_datetime()).days if elo_first_dates.first_elo_dates[2600] is not None else float('nan'),
            "elo2600_to_2700_days": (elo_first_dates.first_elo_dates[2700].year_month_to_datetime() - elo_first_dates.first_elo_dates[2600].year_month_to_datetime()).days if elo_first_dates.first_elo_dates[2700] is not None else float('nan'),
            "max_elo": elo_first_dates.MAX_ELO,
            "max_elo_date": elo_first_dates.MAX_ELO_DATE.year_month_to_datetime() if elo_first_dates.MAX_ELO_DATE is not None else pd.NaT,
            "gm_title_date": elo_first_dates.first_gm_date.year_month_to_datetime() if elo_first_dates.first_gm_date is not None else pd.NaT,
            "max_elo_age": personal_info.get_age_at_datetime(elo_first_dates.MAX_ELO_DATE.year_month_to_datetime()) if personal_info.birthday is not None and elo_first_dates.MAX_ELO_DATE is not None else float('nan'),
            "gm_title_age": personal_info.get_age_at_datetime(elo_first_dates.first_gm_date.year_month_to_datetime()) if personal_info.birthday is not None and elo_first_dates.first_gm_date is not None else float('nan'),
            "2000_2500_avg_abs_elo_change_per_month": elo_first_dates.avg_abs_elo_changes_per_month[2500] if elo_first_dates.avg_abs_elo_changes_per_month[2500] is not None else 0,
            "2000_2600_avg_abs_elo_change_per_month": elo_first_dates.avg_abs_elo_changes_per_month[2600] if elo_first_dates.avg_abs_elo_changes_per_month[2600] is not None else 0,
            "avg_games_per_month_since_2000": elo_first_dates.avg_games_per_month_since_2000 if elo_first_dates.avg_games_per_month_since_2000 is not None else 0,
        }
    
    ### Filter columns
    # Players who were already 2400+ at the time of the first elo report in April 1968
    early_high_elo = base_table["elo2400_date"] < datetime.datetime(1968, 4, 1)

    base_table = base_table[~early_high_elo]    # These players don't have useful data for our models.

    # Players who became GMs after 1990
    ninties_or_later_gms = base_table["gm_title_date"] >= datetime.datetime(1990, 1, 1)
    
    # Players who ever became a gm
    ever_gms = base_table["gm_title_date"].notna()
    gm_and_max_elo_ages_defined = base_table["gm_title_age"].notna() & base_table["max_elo_age"].notna()

    # Male and female players
    male_players = base_table["sex"] == "M"
    female_players = base_table["sex"] == "F"
    
    # Players born after 2000
    born_after_2000 = base_table["birthday"] > datetime.datetime(2000, 1, 1)

    # Players who hit gm <= 15 years of age 
    gm_before_or_at_15 = base_table["gm_title_age"] <= 15

    # Players who hit gm <= 20 years of age
    gm_before_or_at_20 = base_table["gm_title_age"] <= 20

    # Players who hit 2700 elo
    hit_2700 = base_table["elo2700_date"].notna()

    # Players who didn't hit 2700 elo, but were gms
    gm_but_not_2700 = ever_gms & base_table["elo2700_date"].isna()

    # Players who were gms <= 16 years of age and hit 2700 ever.
    gm_before_15_and_2700 = (base_table["gm_title_age"] <= 16) & (base_table["elo2700_date"].notna())       # Note to self: Not sure why these are different, but mimics paper.

    # Players who were gms <= 20 years of age and hit 2700 ever.
    gm_before_20_and_2700 = (base_table["gm_title_age"] <= 20) & (base_table["elo2700_date"].notna())

    models = [
        TabularModelData("Paper Model 1", base_table[hit_2700], X_columns=["elo2500_to_2600_days"], Y_columns=["elo2600_to_2700_days"]),
        TabularModelData("Paper Model 1 Bryce Modification 1", base_table[hit_2700], X_columns=["elo2400_to_2500_days", "2000_2600_avg_abs_elo_change_per_month"], Y_columns=["elo2600_to_2700_days"]),
        TabularModelData("Paper Model 1 Bryce Modification 2", base_table[hit_2700], X_columns=["elo2400_to_2500_days", "avg_games_per_month_since_2000"], Y_columns=["elo2600_to_2700_days"]),
        TabularModelData("Paper Model 1 Bryce Modification 1+2", base_table[hit_2700], X_columns=["elo2400_to_2500_days", "2000_2600_avg_abs_elo_change_per_month", "avg_games_per_month_since_2000"], Y_columns=["elo2600_to_2700_days"]),
        TabularModelData("Paper Model 2", base_table[hit_2700 & ninties_or_later_gms], X_columns=["elo2400_to_2500_days", "elo2500_to_2600_days"], Y_columns=["elo2600_to_2700_days"]),     # Unable to reproduce n from the study.
        TabularModelData("Paper Model 3", base_table[gm_and_max_elo_ages_defined], X_columns=["gm_title_age"], Y_columns=["max_elo_age"]),
        TabularModelData("Paper Model 4", base_table[gm_and_max_elo_ages_defined & male_players], X_columns=["gm_title_age"], Y_columns=["max_elo_age"]),
        TabularModelData("Paper Model 5", base_table[gm_and_max_elo_ages_defined & female_players], X_columns=["gm_title_age"], Y_columns=["max_elo_age"]),
        TabularModelData("Paper Model 6", base_table[gm_before_or_at_15], X_columns=["gm_title_age"], Y_columns=["max_elo_age"]),
        TabularModelData("Paper Model 7", base_table[gm_before_or_at_20], X_columns=["gm_title_age"], Y_columns=["max_elo_age"]),
        TabularModelData("Paper Model 8", base_table[hit_2700], X_columns=["gm_title_age"], Y_columns=["max_elo_age"]),
        TabularModelData("Paper Model 9", base_table[gm_but_not_2700 & gm_and_max_elo_ages_defined], X_columns=["gm_title_age"], Y_columns=["max_elo_age"]),
        TabularModelData("Paper Model 10", base_table[born_after_2000 & ever_gms & gm_and_max_elo_ages_defined], X_columns=["gm_title_age"], Y_columns=["max_elo_age"]),
        TabularModelData("Paper Model 11", base_table[gm_before_15_and_2700 & gm_and_max_elo_ages_defined], X_columns=["gm_title_age"], Y_columns=["elo2600_to_2700_days"]),
        TabularModelData("Paper Model 12", base_table[gm_before_20_and_2700 & gm_and_max_elo_ages_defined], X_columns=["gm_title_age"], Y_columns=["elo2600_to_2700_days"]),
    ]

    condition_sets = {}
    if return_condition_sets_and_personal_info:
        condition_sets["Model1"] = set(base_table[hit_2700]["fideid"])
        condition_sets["Model2"] = set(base_table[hit_2700 & ninties_or_later_gms]["fideid"])
        condition_sets["Model3"] = set(base_table[gm_and_max_elo_ages_defined]["fideid"])
        condition_sets["Model4"] = set(base_table[gm_and_max_elo_ages_defined & male_players]["fideid"])
        condition_sets["Model5"] = set(base_table[gm_and_max_elo_ages_defined & female_players]["fideid"])
        condition_sets["Model6"] = set(base_table[gm_before_or_at_15]["fideid"])
        condition_sets["Model7"] = set(base_table[gm_before_or_at_20]["fideid"])
        condition_sets["Model8"] = set(base_table[hit_2700]["fideid"])
        condition_sets["Model9"] = set(base_table[gm_but_not_2700 & gm_and_max_elo_ages_defined]["fideid"])
        condition_sets["Model10"] = set(base_table[born_after_2000 & ever_gms & gm_and_max_elo_ages_defined]["fideid"])
        condition_sets["Model11"] = set(base_table[gm_before_15_and_2700 & gm_and_max_elo_ages_defined]["fideid"])
        condition_sets["Model12"] = set(base_table[gm_before_20_and_2700 & gm_and_max_elo_ages_defined]["fideid"])

        return condition_sets, personal_infos, elo_first_dates_dict

    return models

def get_full_timeseries_model(elo_cutoff):
    player_data = open_filtered_standard_data(elo_threshold=elo_cutoff)

    players_per_condition, personal_infos, elo_first_dates_dict = build_tables_for_paper_models(player_data, return_condition_sets_and_personal_info=True)

    model_tables = {
        f"Model{model_num}": pd.DataFrame(columns=["fideid", "time", "age_at_time", "elo", "games"]) for model_num in range(1, 13)
    }

    model_tables[f"All ever > {elo_cutoff} players"] = pd.DataFrame(columns=["fideid", "time", "age_at_time", "elo", "games"])

    # Now, let's build out a pandas table for each model.
    for fideid, player_months in player_data.items():
        player_info = personal_infos[fideid]
        elo_firsts = elo_first_dates_dict[fideid]

        player_in_model = {i: fideid in players_per_condition[f"Model{i}"] for i in range(1, 13)}

        for year_month, row in player_months.items():
            if player_info.birthday_datetime is None:
                break

            # Unfortunately the best way to do this is a giant set of if statements.
            if row["rating"] is None or row["games"] is None:
                continue

            rating = row["rating"]
            games = row["games"]
            time = year_month.year_month_to_datetime()

            # We always add data to the all players table.
            pandas_row = {
                    "fideid": fideid,
                    "time": time,
                    "age_at_time": player_info.get_age_at_datetime(time),
                    "elo": rating,
                    "games": games,
            }
            
            model_tables[f"All ever > {elo_cutoff} players"].loc[len(model_tables[f"All ever > {elo_cutoff} players"])] = pandas_row

            # For now we'll just do model 3
            if player_in_model[3] and year_month <= elo_firsts.first_gm_date:
                model_tables["Model3"].loc[len(model_tables["Model3"])] = pandas_row
    return model_tables
                              
class PlayerPersonalInformation:
    __slots__ = ['fideid', 'name', 'country', 'sex', 'birthday', 'birthday_datetime']

    def __init__(self, fideid, name, country, sex, birthday):
        self.fideid = fideid
        self.name = name
        self.country = country
        self.sex = sex
        self.birthday = birthday
        self.birthday_datetime = birthday_string_to_datetime(birthday) if birthday is not None else None
    
    def get_age_at_datetime(self, dt):
        if self.birthday_datetime is None:
            return float('nan')
        
        years_passed = dt.year - self.birthday_datetime.year
        if (dt.month, dt.day) < (self.birthday_datetime.month, self.birthday_datetime.day):
            years_passed -= 1
        years_passed += dt.timetuple().tm_yday / 366  # Technically a slight underestimation for most players, but avoids leap year issues.
        return years_passed

if __name__ == "__main__":    
    filtered_data = open_filtered_standard_data()
    #tables_for_models = build_tables_for_paper_models(filtered_data)
    print(get_full_timeseries_model(filtered_data)["Model3"].describe())