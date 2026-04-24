import os
import pathlib
import requests

def download_fide_standard_data():
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    for year in range(2015, 2026): # We download extra data, then filter down later.
        standard_data_path = pathlib.Path('data/standard')
        standard_data_path.mkdir(parents=True, exist_ok=True)

        for month in months:
            if year == 2015 and month == 'jan':
                continue  # Fide data begins in February 2015
        
            url = f'https://ratings.fide.com/download/standard_{month}{str(year)[2:]}frl_xml.zip'
            file_name = f'standard_{month}{str(year)[2:]}frl_xml.zip'
            file_path = standard_data_path / file_name

            if not file_path.exists():
                response = requests.get(url, allow_redirects=False)
                response.raise_for_status()

                with open(file_path, 'wb') as file:
                    file.write(response.content)
                    print(f'Downloaded: {file_name}, Size: {len(response.content)} bytes')

def unpack_fide_standard_data():
    import zipfile

    standard_data_path = pathlib.Path('data/standard')
    xml_files = set(standard_data_path.glob('*.xml'))

    for zip_file in standard_data_path.glob('*.zip'):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            file_names = zip_ref.namelist()
            if all(f in xml_files for f in file_names):
                continue
            zip_ref.extractall(standard_data_path)
            print(f'Unpacked: {zip_file.name}')



def download_and_unpack_fide_standard_data():
    download_fide_standard_data()
    unpack_fide_standard_data()

if __name__ == "__main__":
    download_and_unpack_fide_standard_data()