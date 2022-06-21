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
        self.start_urls = [f'https://m2.ru/moskva/nedvizhimost/kupit-kvartiru/?rooms=studiya&rooms=1-komnata&rooms=2-komnaty&rooms=3-komnaty&rooms=4-komnaty&rooms=5-komnat_i_bolee&rooms=svobodnaya-planirovka&pageNumber={i}' for i in range(1, 201)]

    def start_requests(self):
        for url in self.start_urls:
            yield Request(url, headers={'user-agent': "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36"})

    def parse(self, response: HtmlResponse):
        links = response.xpath("//div[@data-test='offer-title']/a")
        title = response.xpath("//div[@class='LayoutSnippet__title']/a/div[2]/text()").get()
        try:
            rooms = int(re.findall("\d", title.split(", ")[1])[0])
        except Exception:
            rooms = 0

        for link in links:
            yield response.follow(link, callback=self.parse_flat, cb_kwargs={'rooms': rooms})

    def parse_flat(self, response, rooms):

        title = response.xpath("//h1[@itemprop='name']/text()").get()
        okrug = response.xpath("//span[2]/a[@data-test='offer-address']/text()").get()

        metro = response.xpath("//span[@class='SubwayStation']/span[2]/a/text()").get()

        route_time = int(response.xpath("//div[@class='OfferRouteTimeCard']/div[2]/text()").get())

        price = response.xpath("//span[@itemprop='price']/text()").get()
        price = float(price.replace("\xa0", ""))

        try:
            distance_from_center = response.xpath('//div[@class="ClClickableAddress__links"]/span[6]/text()').getall()
            distance_from_center = float(".".join(re.findall("\d+", distance_from_center[3])))
        except IndexError:
            distance_from_center = response.xpath('//div[@class="ClClickableAddress__links"]/span[5]/text()').getall()
            distance_from_center = float(".".join(re.findall("\d+", distance_from_center[3])))

        except ValueError:
            distance_from_center = 2

        total_area = float(".".join(re.findall("\d+", title.split(' ')[3])))

        yield ParserItem(okrug=okrug, metro=metro, distance_from_center=distance_from_center, route_time=route_time,
                         total_area=total_area, rooms=rooms, price=price)

