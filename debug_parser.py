from scrapy.crawler import CrawlerProcess
from scrapy.settings import Settings

from parser import settings
from parser.spiders.m2 import M2Spider

if __name__ == "__main__":
    crawler_settings = Settings()
    crawler_settings.setmodule(settings)

    process = CrawlerProcess(settings=crawler_settings)
    process.crawl(M2Spider)

    process.start()