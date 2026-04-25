import pathlib
import xml.etree.ElementTree as ET
import pickle

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
    def parse_from_filename(file_name: pathlib.Path):
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
    
    def short_form(self) -> str:
        month_str = [k for k, v in months.items() if v == self.month][0]
        year_str = str(self.year)[2:]
        return f"{month_str}{year_str}"
    
    def __repr__(self) -> str:
        return f"YearMonth(year={self.year}, month={self.month})"
    
    def __hash__(self) -> int:
        return hash((self.year, self.month))

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

    return players

def parse_old_format_monthly_data(file_path, year_month: YearMonth):
    # The old FIDE data format is very odd, we have to rely on the spacing of the header line to parse the data.
    FIELDS = ["ID_NUMBER", "NAME", "TITLE", "COUNTRY", year_month.short_form().upper(), "GAMES", "BIRTHDAY", "FLAG"]

    with open(file_path, 'r') as f:
        lines = f.readlines()
    header_line = lines[0]
    fields_starts = [header_line.find(field) for field in FIELDS]

    players = {}

    for line in lines[1:]:
        if line.strip() == "":
            continue
        
        player_id = int(line[fields_starts[0]:fields_starts[1]].strip())
        name = line[fields_starts[1]:fields_starts[2]].strip()
        title = line[fields_starts[2]:fields_starts[3]].strip()
        country = line[fields_starts[3]:fields_starts[4]].strip()
        elo = int(line[fields_starts[4]:fields_starts[5]].strip())
        games = int(line[fields_starts[5]:fields_starts[6]].strip())
        birthday = line[fields_starts[6]:fields_starts[7]].strip().replace('.', '/')
        flag = line[fields_starts[7]:].strip()

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

def load_all_standard_monthly_data():
    standard_data_path = pathlib.Path('data/standard')

    all_data = {}
    players_to_year_months = {}

    for xml_file in standard_data_path.glob('*.xml'):
        year_month = YearMonth.parse_from_filename(xml_file)
        players = parse_new_format_monthly_data(xml_file)
        all_data[year_month] = players

        for player in players:
            if player not in players_to_year_months:
                players_to_year_months[player] = [year_month]
            else:
                players_to_year_months[player].append(year_month)
        
        print("Loaded data for: ", year_month)

    return all_data, players_to_year_months

class UntransformedDataset:
    def __init__(self, all_data, players_to_year_months):
        self.all_data = all_data
        self.players_to_year_months = players_to_year_months

    def __repr__(self) -> str:
        return f"UntransformedDataset(num_year_months={len(self.all_data)}, num_players={len(self.players_to_year_months)})"

def filter_data_by_player_top_elo_and_gm_status(all_data, players_to_year_months):
    remaining_players = set()
    for player, year_months in players_to_year_months.items():
        max_elo = max(all_data[ym][player].rating for ym in year_months)
        is_gm = any(all_data[ym][player].title == 'GM' for ym in year_months)

        if max_elo >= 2500 or is_gm: # The paper only models players with elos above 2500, and GMs.
            remaining_players.add(player)
    
    for ym in all_data:
        all_data[ym] = {player: data for player, data in all_data[ym].items() if player in remaining_players}
    players_to_year_months = {player: yms for player, yms in players_to_year_months.items() if player in remaining_players}
    
    return all_data, players_to_year_months

def filter_and_save_standard_data(path):
    all_data, players_to_year_months = load_all_standard_monthly_data()
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

if __name__ == "__main__":
    print(open_filtered_standard_data())