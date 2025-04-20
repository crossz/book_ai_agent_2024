import requests
from bs4 import BeautifulSoup
import json
import time
import random

class BilibiliCommentCrawler:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.bilibili.com'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def get_video_oid(self, bvid):
        """获取视频的 oid（评论 ID）"""
        url = f'https://api.bilibili.com/x/web-interface/view?bvid={bvid}'
        response = self.session.get(url)
        data = response.json()
        if data['code'] == 0:
            return data['data']['aid']
        return None

    def get_comments(self, oid, page=1):
        """获取指定页数的评论"""
        url = f'https://api.bilibili.com/x/v2/reply/main'
        params = {
            'oid': oid,
            'type': 1,
            'next': page,
            'mode': 3,
            'plat': 1
        }
        response = self.session.get(url, params=params)
        return response.json()

    def crawl_all_comments(self, bvid):
        """爬取视频的所有评论"""
        oid = self.get_video_oid(bvid)
        if not oid:
            print("无法获取视频信息")
            return

        all_comments = []
        page = 0
        while True:
            page += 1
            print(f"正在爬取第 {page} 页评论...")
            data = self.get_comments(oid, page)
            
            if data['code'] != 0:
                print(f"获取评论失败: {data['message']}")
                break

            replies = data['data']['replies']
            if not replies:
                break

            for reply in replies:
                comment = {
                    'user': reply['member']['uname'],
                    'content': reply['content']['message'],
                    'time': reply['ctime'],
                    'like': reply['like']
                }
                all_comments.append(comment)

            # 随机延迟，避免请求过于频繁
            time.sleep(random.uniform(1, 3))

        return all_comments

    def save_comments(self, comments, filename='comments.json'):
        """保存评论到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(comments, f, ensure_ascii=False, indent=2)
        print(f"评论已保存到 {filename}")

def main():
    # 示例：爬取视频 BV1xx411c7mD 的评论
    bvid = input("请输入要爬取的视频 BV 号（例如：BV1xx411c7mD）：")
    crawler = BilibiliCommentCrawler()
    comments = crawler.crawl_all_comments(bvid)
    if comments:
        crawler.save_comments(comments)
        print(f"共爬取到 {len(comments)} 条评论")

if __name__ == '__main__':
    main() 