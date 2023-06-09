﻿# celebrity-face-recognizer
## Install missing package
`pip install -r requirements.txt`

### Run app
`python ./app/app.py`

`flask run --port=5000 --host=0.0.0.0`

### GET _/hello_
`{"status": true}`

### POST _/image_
[   
    {
        "id": "CE004",   
        "name": "Mỹ Tâm",   
        "description": "Ca sĩ",   
        "wiki_url": "https://vi.wikipedia.org/wiki/M%E1%BB%B9_T%C3%A2m",    
        "article": "Phan Thị Mỹ Tâm, thường được biết đến với nghệ danh Mỹ Tâm, là một nữ ca sĩ kiêm sáng tác nhạc, đạo diễn, diễn viên và giám khảo truyền hình người Việt Nam. "       
    },   
    {
        "id": "CE048",    
        "name": "Ngô Kiến Huy",    
        "description": "Ca sĩ",    
        "wiki_url": "https://vi.wikipedia.org/wiki/Ng%C3%B4_Ki%E1%BA%BFn_Huy",     
        "article": "Lê Thành Dương, hay còn được biết đến với nghệ danh Ngô Kiến Huy, là một nam ca sĩ, diễn viên và người dẫn chương trình người Việt Nam. Anh còn được gọi là Bắp, bắt nguồn từ họ của mình. "     
    }    
]

### TEST endpoint
curl localhost:5000/image -F file=@abc.jpg    

## Server Config

### create my_flask_app.service
`sudo nano /etc/systemd/system/celebrity-face-recognizer.service`

### Add content
[Unit]    
Description=My Flask App - celebrity-face-recognizer          
After=network.target       

[Service]      
User=trung     
WorkingDirectory=/home/trung/celebrity-face-recognizer       
ExecStart=/usr/bin/python3 -m flask run --host=0.0.0.0 --port=5000     
Restart=always       

[Install]      
WantedBy=multi-user.target      


### Restart
`sudo systemctl daemon-reload`

### Start
`sudo systemctl start celebrity-face-recognizer.service`

### Start on Boot
`sudo systemctl enable celebrity-face-recognizer.service`


