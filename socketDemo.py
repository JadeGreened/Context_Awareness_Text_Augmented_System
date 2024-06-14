import time

import requests
from urllib.parse import unquote_plus, parse_qs

# 基本URL
BASE_URL = "http://212.64.10.189:80"

def add_message_to_queue(queue_name, message):
    """ 向指定队列添加消息 """
    url = f"{BASE_URL}/{queue_name}"
    response = requests.post(url, data=message)
    print(f"Adding to {queue_name}: {message} - Status Code: {response.status_code}")


def get_data_from_server(queue_name):
    url = f"http://212.64.10.189/{queue_name}"  # 示例URL，替换为实际的URL
    response = requests.get(url)

    if response.status_code == 200:
        # 假设数据以 application/x-www-form-urlencoded 形式返回
        # 首先解码URL编码
        decoded_response = unquote_plus(response.text)

        # 如果确实需要解析键值对
        parsed_data = parse_qs(decoded_response)

        # 假设你期望得到 'message' 键的数据
        if 'message' in parsed_data:
            message = parsed_data['message'][0]  # 获取列表中的第一个元素
            print(f"Received message: {message}")
        else:
            print("No message key found in the response.")
    else:
        print(f"Failed to fetch data: {response.status_code}")


def clear_queue(queue_name):
    """ 清空指定队列 """
    url = f"{BASE_URL}/clear/{queue_name}"
    response = requests.post(url)
    print(f"Clearing {queue_name} - Status Code: {response.status_code}")

def main():
    clear_queue("unity")
    clear_queue("python")
    while True:
        # 测试Unity队列
        get_data_from_server("unity")
        # 测试Python队列
        add_message_to_queue("python", "Hello from Python!!!")  # 应该显示NoData或空
        time.sleep(2)




if __name__ == "__main__":
    main()
