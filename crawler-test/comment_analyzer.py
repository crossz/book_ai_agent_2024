import json
import jieba
import pandas as pd
import matplotlib.pyplot as plt
from snownlp import SnowNLP
from wordcloud import WordCloud
import seaborn as sns
from collections import Counter
from datetime import datetime
import numpy as np
import re

class CommentAnalyzer:
    def __init__(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.comments = json.load(f)
        self.df = pd.DataFrame(self.comments)
        
        # 转换时间戳为日期时间
        self.df['datetime'] = pd.to_datetime(self.df['time'], unit='s')
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS
        plt.rcParams['axes.unicode_minus'] = False

    def get_sentiment_score(self, text):
        """获取文本的情感得分"""
        try:
            return SnowNLP(text).sentiments
        except:
            return 0.5

    def analyze_sentiments(self):
        """分析评论情感并可视化"""
        # 计算情感得分
        self.df['sentiment'] = self.df['content'].apply(self.get_sentiment_score)
        
        plt.figure(figsize=(12, 6))
        plt.hist(self.df['sentiment'], bins=50, color='skyblue', alpha=0.7)
        plt.title('评论情感分布')
        plt.xlabel('情感得分 (0=消极, 1=积极)')
        plt.ylabel('评论数量')
        plt.savefig('sentiment_distribution.png')
        plt.close()

        # 计算平均情感得分
        avg_sentiment = self.df['sentiment'].mean()
        print(f"平均情感得分: {avg_sentiment:.3f}")

    def generate_wordcloud(self):
        """生成词云图"""
        # 分词并过滤停用词
        stop_words = set(['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'])
        
        text = ' '.join(self.df['content'])
        words = jieba.cut(text)
        words = [word for word in words if word not in stop_words and len(word) > 1]
        
        # 生成词云
        wordcloud = WordCloud(
            # font_path='/System/Library/Fonts/PingFang.ttc',  # macOS 中文字体
            font_path='/System/Library/Fonts/STHeiti Light.ttc',  # macOS 中文字体
            width=1200,
            height=800,
            background_color='white'
        ).generate(' '.join(words))
        
        plt.figure(figsize=(15, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('评论词云图')
        plt.savefig('wordcloud.png')
        plt.close()

    def analyze_time_distribution(self):
        """分析评论时间分布"""
        # 按小时统计评论数量
        self.df['hour'] = self.df['datetime'].dt.hour
        hourly_counts = self.df['hour'].value_counts().sort_index()
        
        plt.figure(figsize=(12, 6))
        hourly_counts.plot(kind='bar', color='skyblue', alpha=0.7)
        plt.title('评论时间分布')
        plt.xlabel('小时')
        plt.ylabel('评论数量')
        plt.savefig('time_distribution.png')
        plt.close()

    def analyze_likes_distribution(self):
        """分析点赞数分布"""
        plt.figure(figsize=(12, 6))
        sns.boxplot(y=self.df['like'])
        plt.title('评论点赞数分布')
        plt.ylabel('点赞数')
        plt.savefig('likes_distribution.png')
        plt.close()

        # 输出点赞数统计信息
        likes_stats = self.df['like'].describe()
        print("\n点赞数统计:")
        print(likes_stats)

    def analyze_user_activity(self):
        """分析用户活跃度"""
        user_counts = self.df['user'].value_counts()
        
        plt.figure(figsize=(12, 6))
        user_counts.head(20).plot(kind='bar')
        plt.title('最活跃用户TOP20')
        plt.xlabel('用户名')
        plt.ylabel('评论数量')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('user_activity.png')
        plt.close()

    def analyze_sentiment_by_likes(self):
        """分析点赞数与情感的关系"""
        plt.figure(figsize=(12, 6))
        plt.scatter(self.df['sentiment'], self.df['like'], alpha=0.5)
        plt.title('情感得分与点赞数的关系')
        plt.xlabel('情感得分')
        plt.ylabel('点赞数')
        plt.savefig('sentiment_vs_likes.png')
        plt.close()

    def run_all_analyses(self):
        """运行所有分析"""
        print("开始分析评论...")
        
        print("\n1. 分析情感分布")
        self.analyze_sentiments()
        
        print("\n2. 生成词云图")
        self.generate_wordcloud()
        
        print("\n3. 分析时间分布")
        self.analyze_time_distribution()
        
        print("\n4. 分析点赞分布")
        self.analyze_likes_distribution()
        
        print("\n5. 分析用户活跃度")
        self.analyze_user_activity()
        
        print("\n6. 分析情感与点赞关系")
        self.analyze_sentiment_by_likes()
        
        print("\n分析完成！所有可视化结果已保存为图片文件。")

def main():
    analyzer = CommentAnalyzer('comments.json')
    analyzer.run_all_analyses()

if __name__ == '__main__':
    main() 