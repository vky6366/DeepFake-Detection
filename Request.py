import requests

url = "http://192.168.177.101:5000/upload"
data = {
    "video_url": "https://drive.google.com/file/d/1evlT6YzQeFKIO5zd2Yv33qDu-7ikEWxs/view?usp=drive_link"

}

response = requests.post(url, data=data)
print(response.json())
