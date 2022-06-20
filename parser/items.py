import scrapy


class ParserItem(scrapy.Item):
    okrug = scrapy.Field()
    metro = scrapy.Field()
    route_time = scrapy.Field()
    total_area = scrapy.Field()
    rooms = scrapy.Field()
    price = scrapy.Field()
