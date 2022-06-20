import scrapy
from scrapy.http import HtmlResponse
from scrapy.http import Request
from parser.items import ParserItem
import re


class M2Spider(scrapy.Spider):
    name = 'm2'
    allowed_domains = ['m2.ru']

    def __init__(self):
        super().__init__()
        self.start_urls = [f'https://m2.ru/moskva/nedvizhimost/kupit-kvartiru/?rooms=studiya&rooms=1-komnata&rooms=2-komnaty&rooms=3-komnaty&rooms=4-komnaty&rooms=5-komnat_i_bolee&rooms=svobodnaya-planirovka&pageNumber={i}' for i in range(1, 100)]

    def start_requests(self):
        for url in self.start_urls:
            yield Request(url, headers={'user-agent': "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36"})

    def parse(self, response: HtmlResponse):
        # links = response.xpath("//div[@data-test='offer-title']/a")
        metros = response.xpath("//a[@data-test='subway-link']/text()").getall()
        titles = response.xpath("//div[@class='LayoutSnippet__title']/a/div[2]/text()").getall()
        okrugs = response.xpath("//div[@class='ClClickableAddress__links']/span[2]/a[@data-test='offer-address']/text()").getall()
        route_times = response.xpath("//div[@class='OfferRouteTimeCard']/div[2]/text()").getall()
        prices = response.xpath("//span[@itemprop='price']/text()").getall()

        route_times = " ".join(route_times).split(".")

        for i in range(len(metros)):
            total_area = float(".".join(re.findall('\d+', titles[i].split(", ")[0].replace('\xa0', ''))))
            try:
                rooms = int(re.findall("\d", titles[i].split(', ')[1])[0])
            except IndexError:
                rooms = 0
            route_time = int(re.findall("\d+", route_times[i])[0])
            current_price = int(prices[i].replace("\xa0", ""))
            yield ParserItem(okrug=okrugs[i], metro=metros[i], route_time=route_time, total_area=total_area, rooms=rooms, price=current_price)

