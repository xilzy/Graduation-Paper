from bs4 import BeautifulSoup

defaultName="line"  #要设置的默认值
path='E:\Project\OpenSource\ByteDance\VisActor\pictogram-bagua-name.svg' #svg图像路径

with open(path, 'r') as f:
    soup = BeautifulSoup(f, 'xml')  # 注意必须使用xml解析器
    for path in soup.find_all('path'):
        # 仅当没有name属性时添加
        if not path.has_attr('name'):
            path['name'] = 'line'
    with open('modified.svg', 'w') as out:
        out.write(str(soup))