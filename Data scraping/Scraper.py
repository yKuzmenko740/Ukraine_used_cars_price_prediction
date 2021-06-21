from typing import List

import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import re


class Scraper:

    """
    Class for scraping https://auto.ria.com/uk/
    Created dataset contains (Brand,Model, Year, Run, City,Benz, L(litres), Transmission, Price_USD)
    """
    _path_for_results = None
    _home_url = None
    __BENZO = {'Бензин', 'Дизель', 'Газ', 'Газ / Бензин', 'Гібрид', 'Електро', 'Інше', 'Газ метан', 'Газ пропан-бутан'}

    def __init__(self, start_url: str, path_for_results: str):
        self.__DATA = []
        if not Scraper.__check_start_url(start_url):
            raise ValueError("no page modifier found in url")
        self._home_url = start_url
        self._path_for_results = path_for_results

    @staticmethod
    def __check_start_url(url: str) -> bool:
        """
        checks we can change page in url
        :param url: url
        """
        pattern = r'page={(.*?)}'
        return True if re.match(pattern, url, re.IGNORECASE) is not None else False

    @staticmethod
    def _get_html(link) -> bs:
        """
        Getting html content of page
        :param link: url
        :return: html content of page
        """
        page = requests.get(link, headers={'Connection': 'close'})
        return bs(page.content, 'html.parser')

    @staticmethod
    def _get_tickets(htmlContent):
        """
        Getting list of tabs with information about the cars
        :param htmlContent: html of page
        :return: list of  tabs with cars
        """
        return htmlContent.find_all('div', class_='content-bar')

    def to_csv(self, file_name: str):
        """
        Converting list of dicts to csv
        :param file_name: path to file
        """
        df = pd.DataFrame(self.__DATA)
        df.to_csv(self._path_for_results + file_name.strip(), index=False)

    def _get_link(self, tab) -> bool:
        """
        Check if it is an auction of car
        :param tab: tab with info about the car
        """
        link = tab.find('a', class_='address', href=True)['href']
        if 'auction' not in link.split('/'):
            return link
        return False

    def __get_brand_model(self, link) -> List[str,str,str]:
        """
        Getting brand model and year of the car
        :param link: link to page with car info
        :return: brand,model and year
        """
        soup_page = Scraper._get_html(link)
        h1 = soup_page.find('h1', class_='head')
        spans = h1.find_all('span')
        brand = spans[0].text
        name = spans[1].text
        year = h1.text.split()[-1]
        return [brand, name, year]

    def parse_range(self, page_range: list):
        """
        Lopping throught page_range and parsing
        :param page_range: range of page numbers to parse
        :return: None
        """
        for i in page_range:
            try:
                d_len_before = len(self.__DATA)
                self.parse_page(i)
                print("*" * 30)
                print("*" * 30)
                print(f"Page number {i + 1} parsed")
                print(f"Cars added ----- {len(self.__DATA) - d_len_before}")
                if i % 100 == 0:
                    print(f'All cars : {len(self.__DATA)}')
            except:
                print("Oops, page failed")
        print(f'{len(page_range)} parsed')

    def parse_page(self, page_num: int):
        """
        Parsing single page
        :param page_num: number of page
        :return: None
        """
        try:
            soup_page = Scraper._get_html(self._home_url.format(page_num))
        except:
            print("Oops, page failed")
            return
        tabs = Scraper._get_tickets(soup_page)
        for tab in tabs:
            link = self._get_link(tab)
            if link:
                page_info = self.__get_brand_model(link)
                ul = tab.find('ul', class_='unstyle characteristic')
                li = ul.find_all('li')
                sub = {'Brand': page_info[0],
                       'Model': page_info[1],
                       'Year': page_info[2],
                       'Run': li[0].text.split()[0] if li[0].text != 'Не вказано' else 'Unknown',
                       'City': li[1].text.split()[0] if li[1].text != 'Не вказано' else 'Unknown',
                       'Benz': None,
                       'L': None,
                       'Transmission': li[3].text.replace(" ", '') if li[3].text.replace(" ",
                                                                                         '') != 'Невказано' else 'Unknown',
                       'Price_USD': tab.find('span', class_='size15').text.replace(" ", "").split('$')[0].replace(
                           u'\xa0',
                           u'')}
                benz = li[2].text.strip().split(",", maxsplit=1)
                if len(benz) < 2:
                    if benz[0] in self.__BENZO:
                        sub['Benz'] = benz[0]
                        sub['L'] = 'Unknown'
                    else:
                        sub['Benz'] = 'Unknown'
                        sub['L'] = benz[0].strip().split()[0]
                else:
                    sub['Benz'] = benz[0]
                    sub['L'] = benz[1].strip().split()[0]
                print(f'Car added : {sub["Brand"]}{sub["Model"]}')
                self.__DATA.append(sub)

    def get_data(self):
        return self.__DATA
