import requests, os, typing, re
from bs4 import BeautifulSoup
from automaticTB.config import rootfolder

def get_websoup(url:str,agent='Mozilla/5.0') -> BeautifulSoup:
    """
    from a url, return a beautiful soup object.
    """
    agent_setting={'User-Agent': agent}
    r=requests.get(url,headers=agent_setting)
    html=r.text
    soup=BeautifulSoup(html,"html.parser")
    return soup


def write_html_to_file(soup: BeautifulSoup, filename:str):
    """
    write soup object to a file
    """
    with open(filename,'w',encoding='utf-8') as f:
        f.write(soup.prettify())


def get_soup_from_file(fn: str) -> BeautifulSoup:
    with open(fn,'r',encoding='utf-8') as f:
        html=f.read()
    return BeautifulSoup(html,"html.parser")


class LocalImage:
    # it provide methods to setup a local image from remote.
    def __init__(self, server_root:str, local_root:str) -> None:
        self.server_root = server_root
        self.local_root = local_root
        if not os.path.exists(self.local_root): os.makedirs(self.local_root)
        self._stored_soup = {}

    def _get_local_filename(self, full_remote_url: str) -> str:
        relative_path = os.path.relpath(full_remote_url, self.server_root)
        return os.path.join(self.local_root, relative_path)

    def _download_local_image(self, url:str, fn:str):
        folder, _ = os.path.split(fn)
        if not os.path.exists(folder):
            os.makedirs(folder)
        url_soup = get_websoup(url)
        write_html_to_file(url_soup, fn)

    def get_soup_from_url(self, url: str) -> BeautifulSoup:
        if url not in self._stored_soup:
            # optimize some speed, maybe not noticable?
            local_full_path = self._get_local_filename(url)
            if not os.path.exists(local_full_path):
                self._download_local_image(url, local_full_path)

            self._stored_soup[url] = get_soup_from_file(local_full_path)
        
        return self._stored_soup[url]


class Bilbao_PointGroups:
    server = "https://www.cryst.ehu.es/"
    local = os.path.join(rootfolder, "sitesymmetry", "Bilbao", "local")
    entrance_url = "https://www.cryst.ehu.es/rep/point.html"

    def __init__(self) -> None:
        # generate a dictionory of urls here
        self.bilbao_image = LocalImage(self.server, self.local)
        self.main_page = self.bilbao_image.get_soup_from_url(self.entrance_url)
        self.group_urls = self._get_urls_for_groups()

    def _get_urls_for_groups(self) -> typing.Dict[str, typing.Dict[str, str]]:
        # one problem of this design is that all group pages will have to be downloaded 
        links = self.main_page.find_all("a", {"class":"green1", "href": re.compile(r"cgi-bin.*")})
        pt_name_url = {}
        for link in links:
            groupname = re.sub(r"\s","",link.text)
            group_url = self.server + link["href"]
            pt_name_url[groupname] = {
                "group_url": group_url
            }
        return pt_name_url

    def _get_operation_page_url_from_group_main_page(self, main_url: str) -> str:
        group_soup = self.bilbao_image.get_soup_from_url(main_url)
        operation_page_relative = group_soup.find_all("a", {"href": re.compile(r".*point_genpos.*")})
        return self.server + operation_page_relative[0]["href"]

    @property
    def group_list(self) -> typing.List[str]:
        return list(self.group_urls.keys())

    def get_group_operation_page_as_soup(self, groupname:str) -> BeautifulSoup:
        if "operation_url" not in self.group_urls[groupname]:
            self.group_urls[groupname]["operation_url"] = \
                self._get_operation_page_url_from_group_main_page(self.group_urls[groupname]["group_url"])
        
        return self.bilbao_image.get_soup_from_url(self.group_urls[groupname]["operation_url"])

    def get_group_main_page_as_soup(self, groupname: str) -> BeautifulSoup:
        url = self.group_urls[groupname]["group_url"]
        return self.bilbao_image.get_soup_from_url(url)


if __name__ == "__main__":
    bpg = Bilbao_PointGroups()