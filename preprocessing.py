import pathlib
import xml.etree.ElementTree as ET

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
    
    def __repr__(self) -> str:
        return f"YearMonth(year={self.year}, month={self.month})"
    
    def __hash__(self) -> int:
        return hash((self.year, self.month))

def parse_monthly_data(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    players = []
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
            birthday = int(player.find('birthday').text)
        except (TypeError, ValueError):
            birthday = 1900  # Default to 1900 if birthday is missing or invalid

        flag = player.find('flag').text

        players.append(PlayerMonth(fideid, name, country, sex, title, w_title, o_title, rating, games, k, birthday, flag))

    return players

def load_all_standard_monthly_data():
    standard_data_path = pathlib.Path('data/standard')

    all_data = {}
    players_to_year_months = {}

    for xml_file in standard_data_path.glob('*.xml'):
        year_month = YearMonth.parse_from_filename(xml_file)
        players = parse_monthly_data(xml_file)
        all_data[year_month] = players
        print("Loaded data for: ", year_month)

    return all_data, players_to_year_months

if __name__ == "__main__":
    print(load_all_standard_monthly_data())