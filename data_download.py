import os
import pathlib
import requests

def write_download_complete_marker():
    marker_path = pathlib.Path('data/standard/download_complete.marker')
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(marker_path, 'w') as marker_file:
        marker_file.write('1')

def is_download_complete():
    marker_path = pathlib.Path('data/standard/download_complete.marker')
    if not marker_path.exists():
        return False
    
    with open(marker_path, 'r') as marker_file:
        content = marker_file.read().strip()
        return content == '1'

def download_fide_standard_data():
    if is_download_complete():
        print("Download already complete. Skipping download step.")
        return

    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    standard_data_path = pathlib.Path('data/standard')
    standard_data_path.mkdir(parents=True, exist_ok=True)

    for year in range(2012, 2026): # We download extra data, then filter down later.
        for month in months:
            if year == 2012 and month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul']:
                continue # These months are in the old format

            url = f'https://ratings.fide.com/download/standard_{month}{str(year)[2:]}frl_xml.zip'
            file_name = f'standard_{month}{str(year)[2:]}frl_xml.zip'
            file_path = standard_data_path / file_name

            if not file_path.exists():
                response = requests.get(url, allow_redirects=False)
                try:
                    response.raise_for_status()
                except requests.HTTPError as e:
                    print(f"Failed to download {file_name}, perhaps not available or non-existent: {e}")
                    continue

                with open(file_path, 'wb') as file:
                    file.write(response.content)
                    print(f'Downloaded: {file_name}, Size: {len(response.content)} bytes')

    for year in range(2000, 2012):
        for month in months:        
            url = f'https://ratings.fide.com/download/{month}{str(year)[2:]}frl.zip'
            file_name = f'{month}{str(year)[2:]}.zip'
            file_path = standard_data_path / file_name

            if not file_path.exists():
                response = requests.get(url, allow_redirects=False)
                try:
                    response.raise_for_status()
                except requests.HTTPError as e:
                    print(f"Failed to download {file_name}, perhaps not available or non-existent: {e}")
                    continue

                with open(file_path, 'wb') as file:
                    file.write(response.content)
                    print(f'Downloaded: {file_name}, Size: {len(response.content)} bytes')
    write_download_complete_marker()

def unpack_fide_standard_data():
    import zipfile

    standard_data_path = pathlib.Path('data/standard')
    existing_files = set(standard_data_path.glob('*.xml')) | set(standard_data_path.glob('*.txt')) | set(standard_data_path.glob('*.TXT'))
    existing_files = set(f.name for f in existing_files)


    for zip_file in standard_data_path.glob('*.zip'):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            file_names = zip_ref.namelist()
            if all(f in existing_files for f in file_names):
                continue
            zip_ref.extractall(standard_data_path)
            print(f'Unpacked: {zip_file.name}')

def download_pre_fide_online_data():
    # Pre 2001 data is only easily available on OlimpBase.
    # First we persist the pages to data/standard/olimpbase (to not spam their servers), then we can parse them.
    pages = {
        "Jun 1967": "https://www.olimpbase.org/Elo/Elo196706e.html",
        "Apr 1968": "https://www.olimpbase.org/Elo/Elo196804e.html",
        "Jan 1969": "https://www.olimpbase.org/Elo/Elo196900e.html",
        "Jan 1970": "https://www.olimpbase.org/Elo/Elo197000e.html",
        "Jan 1971": "https://www.olimpbase.org/Elo/Elo197101e.html",
        "Jul 1971": "https://www.olimpbase.org/Elo/Elo197107e.html",
        "Jul 1972": "https://www.olimpbase.org/Elo/Elo197207e.html",
        "Jul 1973": "https://www.olimpbase.org/Elo/Elo197307e.html",
        "May 1974": "https://www.olimpbase.org/Elo/Elo197405e.html",
        "Jan 1975": "https://www.olimpbase.org/Elo/Elo197501e.html",
        "Jan 1976": "https://www.olimpbase.org/Elo/Elo197601e.html",
        "Jan 1977": "https://www.olimpbase.org/Elo/Elo197701e.html",
        "Jan 1978": "https://www.olimpbase.org/Elo/Elo197801e.html",
        "Jul 1978": "https://www.olimpbase.org/Elo/Elo197807e.html",
        "Jan 1979": "https://www.olimpbase.org/Elo/Elo197901e.html",
        "Jul 1979": "https://www.olimpbase.org/Elo/Elo197907e.html",
        "Jan 1980": "https://www.olimpbase.org/Elo/Elo198001e.html",
        "Jul 1980": "https://www.olimpbase.org/Elo/Elo198007e.html",
        "Jan 1981": "https://www.olimpbase.org/Elo/Elo198101e.html",
        "Jul 1981": "https://www.olimpbase.org/Elo/Elo198107e.html",
        "Jan 1982": "https://www.olimpbase.org/Elo/Elo198201e.html",
        "Jul 1982": "https://www.olimpbase.org/Elo/Elo198207e.html",
        "Jan 1983": "https://www.olimpbase.org/Elo/Elo198301e.html",
        "Jul 1983": "https://www.olimpbase.org/Elo/Elo198307e.html",
        "Jan 1984": "https://www.olimpbase.org/Elo/Elo198401e.html",
        "Jul 1984": "https://www.olimpbase.org/Elo/Elo198407e.html",
        "Jan 1985": "https://www.olimpbase.org/Elo/Elo198501e.html",
        "Jul 1985": "https://www.olimpbase.org/Elo/Elo198507e.html",
        "Jan 1986": "https://www.olimpbase.org/Elo/Elo198601e.html",
        "Jul 1986": "https://www.olimpbase.org/Elo/Elo198607e.html",
        "Jan 1987": "https://www.olimpbase.org/Elo/Elo198701e.html",
        "Jul 1987": "https://www.olimpbase.org/Elo/Elo198707e.html",
        "Jan 1988": "https://www.olimpbase.org/Elo/Elo198801e.html",
        "Jul 1988": "https://www.olimpbase.org/Elo/Elo198807e.html",
        "Jan 1989": "https://www.olimpbase.org/Elo/Elo198901e.html",
        "Jul 1989": "https://www.olimpbase.org/Elo/Elo198907e.html",
        "Jan 1990": "https://www.olimpbase.org/Elo/Elo199001e.html",    # First set with IDs.
        "Jul 1990": "https://www.olimpbase.org/Elo/Elo199007e.html",
        "Jan 1991": "https://www.olimpbase.org/Elo/Elo199101e.html",
        "Jul 1991": "https://www.olimpbase.org/Elo/Elo199107e.html",
        "Jan 1992": "https://www.olimpbase.org/Elo/Elo199201e.html",
        "Jul 1992": "https://www.olimpbase.org/Elo/Elo199207e.html",
        "Jan 1993": "https://www.olimpbase.org/Elo/Elo199301e.html",
        "Jul 1993": "https://www.olimpbase.org/Elo/Elo199307e.html",
        "Jan 1994": "https://www.olimpbase.org/Elo/Elo199401e.html",
        "Jul 1994": "https://www.olimpbase.org/Elo/Elo199407e.html",
        "Jan 1995": "https://www.olimpbase.org/Elo/Elo199501e.html",
        "Jul 1995": "https://www.olimpbase.org/Elo/Elo199507e.html",
        "Jan 1996": "https://www.olimpbase.org/Elo/Elo199601e.html",
        "Jul 1996": "https://www.olimpbase.org/Elo/Elo199607e.html",
        "Jan 1997": "https://www.olimpbase.org/Elo/Elo199701e.html",
        "Jul 1997": "https://www.olimpbase.org/Elo/Elo199707e.html",
        "Jan 1998": "https://www.olimpbase.org/Elo/Elo199801e.html",
        "Jul 1998": "https://www.olimpbase.org/Elo/Elo199807e.html",
        "Jan 1999": "https://www.olimpbase.org/Elo/Elo199901e.html",
        "Jul 1999": "https://www.olimpbase.org/Elo/Elo199907e.html",
        "Jan 2000": "https://www.olimpbase.org/Elo/Elo200001e.html",
        "Jul 2000": "https://www.olimpbase.org/Elo/Elo200007e.html",
        "Oct 2000": "https://www.olimpbase.org/Elo/Elo200010e.html",
        "Jan 2001": "https://www.olimpbase.org/Elo/Elo200101e.html",
        "Apr 2001": "https://www.olimpbase.org/Elo/Elo200104e.html",
        "Jul 2001": "https://www.olimpbase.org/Elo/Elo200107e.html",
        "Oct 2001": "https://www.olimpbase.org/Elo/Elo200110e.html"
    }

    # Saving pages to disk, under data/standard/olimpbase/ELO{date}.html
    olimpbase_path = pathlib.Path('data/standard/olimpbase')
    olimpbase_path.mkdir(parents=True, exist_ok=True)
    for date, url in pages.items():
        file_name = f"ELO{date.replace(' ', '')}.html"
        file_path = olimpbase_path / file_name

        if not file_path.exists():
            response = requests.get(url, allow_redirects=False)
            try:
                response.raise_for_status()
            except requests.HTTPError as e:
                print(f"Failed to download {file_name} from OlimpBase, perhaps not available or non-existent: {e}")
                continue

            with open(file_path, 'wb') as file:
                file.write(response.content)
                print(f'Downloaded: {file_name}, Size: {len(response.content)} bytes')

def download_and_unpack_fide_standard_data():
    download_fide_standard_data()
    unpack_fide_standard_data()
    download_pre_fide_online_data()

if __name__ == "__main__":
    download_and_unpack_fide_standard_data()