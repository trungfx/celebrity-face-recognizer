import json
import os
from icrawler.builtin import GoogleImageCrawler
from tqdm import tqdm

# tắt thông báo lỗi ERROR:downloader:... 404, 400
import logging

logging.disable(logging.CRITICAL)

# load data từ json
with open('../data/json/vietnam_celeb.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for celeb in tqdm(data):
    name = celeb['name']
    _id = celeb['id']
    description = celeb['description']
    article = celeb['article'].split(".")[0]

    # tạo thư mục với tên là _id
    dir_path = os.path.join('../data', 'images_crawl', str(_id))
    os.makedirs(dir_path, exist_ok=True)

    # crawl images và lưu vào thư mục trên
    google_crawler = GoogleImageCrawler(
        feeder_threads=1,
        parser_threads=4,
        downloader_threads=16,
        storage={'root_dir': dir_path}
    )
    filters = dict(
        # type='face',
        type='photo',
        color='color'
    )
    try:
        google_crawler.crawl(
            keyword=description + ' ' + name,
            filters=filters,
            max_num=200,
            min_size=(50, 50),
            file_idx_offset='auto'
        )
    except Exception as e:
        print(f"Error while crawling images for {name}: {e}")
