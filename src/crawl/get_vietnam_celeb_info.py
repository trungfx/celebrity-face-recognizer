import requests
import json
import os


# Lấy danh sách người nổi tiếng từ google
def get_celeb_info(search_query, filename, google_api_key, id_header, result_limit):
    service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
    params = {
        'query': search_query,
        'limit': result_limit,
        'types': ['Person'],
        'indent': True,
        'languages': 'vi',
        'key': google_api_key,
    }
    response = requests.get(service_url, params=params)
    data = response.json()

    results = []
    for item in data['itemListElement']:
        # Lấy các thông tin cân thiết từ kết quả trả về
        name = item['result'].get('name', '')
        description = item['result'].get('description', '')
        wiki_url = item['result'].get('detailedDescription', {}).get('url', '')
        article = item['result'].get('detailedDescription', {}).get('articleBody', '')

        if name and description and wiki_url and article:
            i = len(results) + 1
            ids = id_header + '{:03d}'.format(i)
            result = {
                'id': ids,
                'name': name,
                'description': description,
                'wiki_url': wiki_url,
                'article': article
            }
            results.append(result)

    # Lưu vào file json
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Đã lưu thông tin của {len(results)} '{search_query}' vào file {filename}.")


api_key = open('.api_key').read()
if not os.path.exists('../data/json'):
    os.makedirs('../data/json')

get_celeb_info('người việt nam nổi tiếng', '../data/json/vietnam_celeb.json', api_key, 'CE', 200)
