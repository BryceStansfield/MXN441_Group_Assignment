import pathlib
from urllib import response
import xml.etree.ElementTree as ET
import pickle
import bs4
import multiprocessing

months = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
          'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

"""
<player>
<fideid>35077023</fideid>
<name>A Chakravarthy</name>
<country>IND</country>
<sex>M</sex>
<title></title>
<w_title></w_title>
<o_title></o_title>
<rating>1151</rating>
<games>0</games>
<k>40</k>
<birthday>1986</birthday>
<flag></flag>
</player>

"""
class PlayerMonth:
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
    
    def __repr__(self) -> str:
        return f"YearMonth(year={self.year}, month={self.month})"
    
    def __hash__(self) -> int:
        return hash((self.year, self.month))

    def __lt__(self, other):
        if self.year == other.year:
            return self.month < other.month
        return self.year < other.year

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

def parse_all_old_format_monthly_data(standard_data_path):
    all_data = {}
    all_files = list(standard_data_path.glob('*.TXT')) + list(standard_data_path.glob('*.txt'))
    year_months = [YearMonth.parse_from_old_fide_filename(file) for file in all_files]
    players_per_file = [parse_old_format_monthly_data(file, ym) for file, ym in zip(all_files, year_months)]

    all_data = {ym: players for ym, players in zip(year_months, players_per_file)}

    return all_data

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

def load_all_olimpbase_monthly_data():
    all_data = {}

    for file in pathlib.Path('data/standard/olimpbase').glob('*.html'):
        year_month = YearMonth.parse_from_olimpbase_filename(file) # Bit hacky
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
        
    return all_data
    
def load_all_standard_monthly_data():
    standard_data_path = pathlib.Path('data/standard')
    files = list(standard_data_path.glob('*.xml'))
    year_months = [YearMonth.parse_from_new_fide_filename(file) for file in files]
    players_per_file = [parse_new_format_monthly_data(file) for file in files]
    
    all_data = {ym: players for ym, players in zip(year_months, players_per_file)}        

    return all_data

class UntransformedDataset:
    def __init__(self, all_data, players_to_year_months):
        self.all_data = all_data
        self.players_to_year_months = players_to_year_months

    def __repr__(self) -> str:
        return f"UntransformedDataset(num_year_months={len(self.all_data)}, num_players={len(self.players_to_year_months)})"

def filter_data_by_player_top_elo_and_gm_status(all_data, players_to_year_months):
    remaining_players = set()
    for player, year_months in players_to_year_months.items():
        max_elo = max(all_data[ym][player].rating if all_data[ym][player].rating is not None else 0 for ym in year_months)
        is_gm = any(all_data[ym][player].title == 'GM' for ym in year_months)

        if max_elo >= 2500 or is_gm: # The paper only models players with elos above 2500, and GMs.
            remaining_players.add(player)
    
    for ym in all_data:
        all_data[ym] = {player: data for player, data in all_data[ym].items() if player in remaining_players}
    players_to_year_months = {player: yms for player, yms in players_to_year_months.items() if player in remaining_players}
    
    return all_data, players_to_year_months

def filter_and_save_standard_data(path):
    olimp_data = load_all_olimpbase_monthly_data()
    all_data = {}
    old_fide_data = parse_all_old_format_monthly_data(pathlib.Path('data/standard'))
    all_data = load_all_standard_monthly_data()
    all_data.update(old_fide_data)
    all_data.update(olimp_data)

    players_to_year_months = {}
    for ym, players in all_data.items():
        for player in players:
            if player not in players_to_year_months:
                players_to_year_months[player] = set()
            players_to_year_months[player].add(ym)

    # Temp, for debugging
    with open(pathlib.Path('data/standard/players_to_year_months.pkl'), 'wb') as f:
        pickle.dump(players_to_year_months, f)

    all_data, players_to_year_months = filter_data_by_player_top_elo_and_gm_status(all_data, players_to_year_months)

    untransformed_data = UntransformedDataset(all_data, players_to_year_months)
    with open(path, 'wb') as f:
        pickle.dump(untransformed_data, f)
    return untransformed_data

def open_filtered_standard_data():
    filtered_data_path = pathlib.Path('data/standard/filtered_standard_data.pkl')
    if filtered_data_path.exists():
        with open(filtered_data_path, 'rb') as f:
            return pickle.load(f)
    else:
        return filter_and_save_standard_data(pathlib.Path('data/standard/filtered_standard_data.pkl'))

class PlayerFirstEloDates:
    def __init__(self, player_id, player_data_by_year_month):
        self.player_to_first_elo_date = player_to_first_elo_date

if __name__ == "__main__":
    print(open_filtered_standard_data())