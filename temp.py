from bs4 import BeautifulSoup,NavigableString
from pprint import pprint
file_path = 'sample.html'  # HTML 파일의 경로

# HTML 파일을 읽어옴
with open(file_path, 'r', encoding='utf-8') as file:
    html_content = file.read()

# BeautifulSoup를 사용하여 HTML 파싱
soup = BeautifulSoup(html_content, 'html.parser')


# Extract specific elements from the HTML

title = soup.title
h1 = soup.h1
p = soup.p
ul = soup.ul

# Print the text content of the extracted elements
# print("Title:", title.text)
# print("H1:", h1.text)
# print("Paragraph:", p.text)
# print("List items:")
# for li in ul.find_all('li'):
#     print(li.text)
for node in soup.find_all('html')[0]:
    node = node.children
    print("NODE: ", node)
    if isinstance(node, NavigableString):
        print("YES", node)
    else:
        print("NO", node)
        
