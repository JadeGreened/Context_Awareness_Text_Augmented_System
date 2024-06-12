import requests

# 设定你的Spring Boot应用的URL
BASE_URL = "http://212.64.10.189"


def send_data(message):
    """发送数据到队列"""
    url = f"{BASE_URL}/sendData"
    response = requests.post(url, json=message)
    return response.text


def get_data():
    """从队列获取数据"""
    url = f"{BASE_URL}/getData"
    response = requests.get(url)
    return response.text


def clear_queue():
    """清空队列"""
    url = f"{BASE_URL}/clear"
    response = requests.post(url)
    return response.text


if __name__ == "__main__":
    send_data("Hello World")
    print(get_data())
