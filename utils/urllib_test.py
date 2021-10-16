import urllib.parse as parser
import requests as requester
from lxml import html as html_parser
from lxml import etree
from typing import List


class HtmlCrawl:

    def crawl(self, url: str) -> str:
        url_str = '%B4%F3%CA%FD%BE%DD&key3=%C7%E5%BB%AA%B4%F3%D1%A7%B3%F6%B0%E6%C9%E7'
        unencode_url = parser.unquote(url_str, 'utf-8')  # 解码
        print(unencode_url)

        headers = {
            'authority': 'www.zhihu.com',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36 Edg/88.0.705.68',
            'x-requested-with': 'fetch',
            'content-type': 'multipart/form-data; boundary=----WebKitFormBoundarycwskcLmf85lBwPKR',
            'accept': '*/*',
            'origin': 'https://www.zhihu.com',
            'sec-fetch-site': 'same-origin',
            'sec-fetch-mode': 'cors',
            'sec-fetch-dest': 'empty',
            'referer': 'https://www.zhihu.com/',
            'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
        }

        response = requester.get(url, headers=headers)
        content = response.content
        # f.write(content)
        return str(content, 'utf-8')

    def read(self, file_name: str):
        f = open(file_name, "r")
        text = f.read()
        root = html_parser.fromstring(text)
        image_urls = root.xpath('//img[@data-original]/@data-original')
        for index, img in enumerate(image_urls):
            print('img:', img)
            # self.crawl(img, 'html/' + str(index) + '.jpeg')

    def read_div(self, text: str) -> List[str]:
        html_code = etree.HTML(text)
        # ret = html_code.xpath('//div[@class="RichContent-inner"]/p/text()')
        ret = html_code.xpath('//div[@class="RichContent-inner"]/span/*/text()')
        ans = []
        for i, r in enumerate(ret):
            ans.append(r)
            print(i, ':', r)
        return ans

    def parse_url(self, url: str) -> str:
        return url.split('.')[-1]


h = HtmlCrawl()
# content = h.crawl('https://www.zhihu.com/question/31955622/answer/1625152059', "html/1625152059.html")
# content = h.crawl('https://zhuanlan.zhihu.com/p/27624814', "html/runoob_urllib_test.html")
# h.parse_result(content)
# https://www.cnblogs.com/zhangxinqi/p/9210211.html#_label1
# https://zhuanlan.zhihu.com/p/371637064
# h.read_div("/Users/zhangshipeng/PycharmProjects/ephemeralP/html/1625152059.html")
# print(h.parse_url('https://pic4.zhimg.com/v2-5ae91b7cd6224a533b18fc8916eb5513_r.jpg'))

# f = open('/Users/zhangshipeng/PycharmProjects/ephemeralP/html/url.txt', "wb")
f = open('/Users/zhangshipeng/PycharmProjects/ephemeralP/html/url.txt', "r")
line = f.readlines()
name = ''
# ans = []
f2 = open('/Users/zhangshipeng/PycharmProjects/ephemeralP/html/dakuankuan.txt', "wb")
for i, l in enumerate(line):
    if i % 2 == 0:
        name = l
    else:
        uu = l
        print(name[:-1], ':', uu[:-1])
        f2.write(bytes(str(i // 2 + 1) + ':', 'utf-8'))
        f2.write(bytes(name[:-1], 'utf-8'))
        f2.write(bytes('\n', 'utf-8'))
        temp = h.read_div(h.crawl(uu[:-1]))
        for t in temp:
            f2.write(bytes(t, 'utf-8'))
            f2.write(bytes('\n', 'utf-8'))
        f2.write(bytes('\n', 'utf-8'))
        # break
f2.close()
