# 读取表格的对象
# 读取表格中的每一列
# 将表格中的每一列的数据进行排序
# 定义四种颜色的字符
# 创建一个写的表格对象
# 如果是排序第一的数据 就用红色写入，蓝色，黄色，绿色四种颜色
import xlrd
import xlwt

#参数
# readFileName = 'F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-ct\wgif-D2FNet\第一篇\平均值比较.xlsx' # 访问的表格的相对路径
# saveFile = 'F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-ct\wgif-D2FNet\第一篇\平均值比较2.xls' # 生成的文件,xlwt只能写xls后缀的才能打开
# readFileName = 'F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-pet\wgif-D2FNet\第一篇\平均值比较.xlsx' # 访问的表格的相对路径
# saveFile = 'F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-pet\wgif-D2FNet\第一篇\平均值比较2.xls' # 生成的文件,xlwt只能写xls后缀的才能打开
# readFileName = 'F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-spect\wgif-D2FNet\第一篇\平均值比较.xlsx' # 访问的表格的相对路径
# saveFile = 'F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-spect\wgif-D2FNet\第一篇\平均值比较2.xls' # 生成的文件,xlwt只能写xls后缀的才能打开


# readFileName = 'F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-ct\D2FNet\第二篇\平均值比较.xlsx' # 访问的表格的相对路径
# saveFile = 'F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-ct\D2FNet\第二篇\平均值比较2.xls' # 生成的文件,xlwt只能写xls后缀的才能打开
# readFileName = 'F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-pet\D2FNet\第二篇\平均值比较.xlsx' # 访问的表格的相对路径
# saveFile = 'F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-pet\D2FNet\第二篇\平均值比较2.xls' # 生成的文件,xlwt只能写xls后缀的才能打开
# readFileName = 'F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-spect\D2FNet\第二篇\平均值比较.xlsx' # 访问的表格的相对路径
# saveFile = 'F:\论文整理构思\评价指标\傅羽\DATA\方法对比\mri-spect\D2FNet\第二篇\平均值比较2.xls' # 生成的文件,xlwt只能写xls后缀的才能打开

# readFileName = 'E:\论文写作\杨老师\论文投稿\第二篇\材料\表格\对比实验\GFP-PCI均值比较.xlsx' # 访问的表格的相对路径
# saveFile = 'E:\论文写作\杨老师\论文投稿\第二篇\材料\表格\对比实验\GFP-PCI均值比较2.xls' # 生成的文件,xlwt只能写xls后缀的才能打开
# readFileName = 'E:\论文写作\杨老师\论文投稿\第二篇\材料\表格\消融实验\平均值比较.xlsx' # 访问的表格的相对路径
# saveFile = 'E:\论文写作\杨老师\论文投稿\第二篇\材料\表格\消融实验\平均值比较2.xls' # 生成的文件,xlwt只能写xls后缀的才能打开
# readFileName = 'E:\论文写作\杨老师\论文投稿\第二篇\材料\表格\拓展实验\MRI-PET平均值比较.xlsx'
# saveFile = 'E:\论文写作\杨老师\论文投稿\第二篇\材料\表格\拓展实验\MRI-PET平均值比较2.xls'
readFileName = 'E:\论文写作\杨老师\论文投稿\第二篇\材料\表格\拓展实验\MRI-SPECT平均值比较.xlsx'
saveFile = 'E:\论文写作\杨老师\论文投稿\第二篇\材料\表格\拓展实验\MRI-SPECT平均值比较2.xls'
# 创建特别样式，特定的数据按照该样式写入表格之中对应位置
style1 = xlwt.XFStyle() # 初始化样式
font = xlwt.Font() # 为样式创建字体
font.name = 'Times New Roman'
font.bold = True # 加粗
font.colour_index = 2               #红色
style1.font = font
# 创建特别样式，特定的数据按照该样式写入表格之中对应位置
style2 = xlwt.XFStyle() # 初始化样式
font = xlwt.Font() # 为样式创建字体
font.name = 'Times New Roman'
font.bold = True # 加粗
font.colour_index = 4                  #蓝色
style2.font = font
# 创建特别样式，特定的数据按照该样式写入表格之中对应位置
style3 = xlwt.XFStyle() # 初始化样式
font = xlwt.Font() # 为样式创建字体
font.name = 'Times New Roman'
font.bold = True # 加粗
font.colour_index = 52                #黄色
style3.font = font
# 创建特别样式，特定的数据按照该样式写入表格之中对应位置
style4 = xlwt.XFStyle() # 初始化样式
font = xlwt.Font() # 为样式创建字体
font.name = 'Times New Roman'
font.bold = True # 加粗
font.colour_index = 3                      #绿色
style4.font = font


# 打开目标表 并生成对应表的sheet
data = xlrd.open_workbook(readFileName)
table = data.sheets()[0]
# 总的行数
numOfRow = table.nrows
# 总的列数
numOfCol = table.ncols
# 获取第一行的内容
fistRowValue = table.row_values(0)
# 获取第一列的内容
fistColValue = table.col_values(0)
# 第二列的内容
secondColValue = table.col_values(1)
# 创建一个workbook
workbook = xlwt.Workbook(encoding='utf-8')
# 创建一个worksheet
worksheet = workbook.add_sheet('sheet1')
# 将第一列和第一排的数据写入
# 将第一行数据写入
for i in range(len(fistRowValue)):
    worksheet.write(0, i, fistRowValue[i])
for i in range(1,len(fistColValue)):
    worksheet.write(i,0,fistColValue[i])
for i in range(1,len(secondColValue)):
    worksheet.write(i,1,secondColValue[i])

for i in range(2,numOfCol):
    allColValues = table.col_values(i)[1:]
    if table.col_values(i)[0] == 'QCV' or table.col_values(i)[0] == 'OCE':
        theSortValue = sorted(allColValues)
    else:
        theSortValue = sorted(allColValues,reverse=True)
    for j in range(0,len(allColValues)):
        if allColValues[j] == theSortValue[0]:
            worksheet.write(j+1, i, allColValues[j],style1)
        elif allColValues[j] == theSortValue[1]:
            worksheet.write(j+1, i, allColValues[j],style2)
        elif allColValues[j] == theSortValue[2]:
            worksheet.write(j+1, i, allColValues[j],style3)
        elif allColValues[j] == theSortValue[3]:
            worksheet.write(j+1, i, allColValues[j],style4)
        else:
            worksheet.write(j + 1, i, allColValues[j])
workbook.save(saveFile)


