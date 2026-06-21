# ###根据每张图片的指标数据，筛选出所有至少k个指标都大于EMFusion的，输出筛选出来的图像编号的txt文件
# import xlrd      #pip install xlrd==1.2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple,要装低版本的，高版本不支持读xlsx后缀的文件了
#
# ### 参数
# # excel_path1="E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-ct\D2FNet.xlsx"
# # excel_path2="E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-ct\EMFusion.xlsx"
# # excel_path1="E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-pet\D2FNet.xlsx"
# # excel_path2="E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-pet\EMFusion.xlsx"
# excel_path1="E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-spect\D2FNet.xlsx"
# excel_path2="E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-spect\EMFusion.xlsx"
# a=excel_path1.split('\\')
# k=2                             #需要筛选多少个指标优于平均数的图片
# metric_nums=[4,9,11,14]         #要对比的指标的编号
# #print(a[-2])
# txt_name=a[-2]+"_selected_picture_"+str(k)+".txt"
#
# def select(excel_path1,excel_path2,metric_nums,txt_name,k):
#     book1 = xlrd.open_workbook(excel_path1)       #获取工作簿对象
#     sheets1=book1.sheets()                        #获取所有表单
#     sheet1=sheets1[0]                            #获取第一张表单
#
#     book2 = xlrd.open_workbook(excel_path2)       #获取工作簿对象
#     sheets2=book2.sheets()                        #获取所有表单
#     sheet2=sheets2[0]                            #获取第一张表单
#
#     # 获取行数和列数
#     nrows = sheet1.nrows
#     ncols = sheet1.ncols
#     #print(nrows,ncols)                      #185,16
#
#     # #开始计算每一列的平均值
#     # #print(sheet1.col_values(1))
#     # average=[0]                            #用于记录每一列的平均值,第一列是名字，用0占位
#     # for i in range(1,ncols):                  #第一列是序号不读进去
#     #     col_i=sheet1.col_values(i)
#     #     col_i.pop(0)                          #把每一列的第一个中文标题去掉
#     #     average.append(sum(col_i)/(nrows-1))
#     # #print(average)
#
#     nums=[]          #记录被挑选的图片的序号
#     #print(nums)
#     #开始遍历每一张图的每一个指标，来筛选全部指标都大于均值的图片
#     for i in range(1,nrows):                #第一行不要是抬头，后面每一行就是一张图
#         c=0                                 #用于记录一张图片有几个指标大于平均值(最后一个指标是小于平均值才更好)
#         row1_i=sheet1.row_values(i)
#         row2_i = sheet2.row_values(i)
#         for j in metric_nums:
#             if(j!=ncols-1):
#                 if (row1_i[j] >= row2_i[j]):
#                     c += 1
#             else:
#                 if (row1_i[j] <= row2_i[j]):
#                     c += 1
#
#         if(c>=k):                            #如果满足筛选条件，就选
#             nums.append(i)
#     print(sheet1.row_values(0)[4],sheet1.row_values(0)[9],sheet1.row_values(0)[11],sheet1.row_values(0)[14])
#     print(len(nums))
#     print(nums)
#     with open(txt_name,'w') as f:
#         for n in nums:
#             f.write(str(n) + '.png' + '\n')
#
#
#
# if(__name__=='__main__'):
#     select(excel_path1,excel_path2,metric_nums,txt_name,k)


###根据每张图片的指标数据，筛选出选定的指标中，至少k个指标都大于DPCN的，输出筛选出来的图像编号的txt文件
import xlrd      #pip install xlrd==1.2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple,要装低版本的，高版本不支持读xlsx后缀的文件了

### 参数
# excel_path1="E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-ct\D2FNet.xlsx"
# excel_path2="E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-ct\EMFusion.xlsx"
# excel_path1="E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-pet\D2FNet.xlsx"
# excel_path2="E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-pet\EMFusion.xlsx"
excel_path1="F:\论文整理构思\评价指标\傅羽\DATA\方法对比\gfp-pci\proposed\第三篇\每张图片指标.xlsx"
excel_path2="F:\论文整理构思\评价指标\傅羽\DATA\方法对比\gfp-pci\MURF\第三篇\每张图片指标.xlsx"
a=excel_path1.split('\\')
k=8                             #需要筛选多少个指标优于平均数的图片
metric_nums=[1,2,3,5,6,7,8,9,10]         #要对比的指标的编号
#print(a[-2])
txt_name=a[-2]+"_selected_picture_"+str(k)+".txt"

def select(excel_path1,excel_path2,metric_nums,txt_name,k):
    book1 = xlrd.open_workbook(excel_path1)       #获取工作簿对象
    sheets1=book1.sheets()                        #获取所有表单
    sheet1=sheets1[0]                            #获取第一张表单

    book2 = xlrd.open_workbook(excel_path2)       #获取工作簿对象
    sheets2=book2.sheets()                        #获取所有表单
    sheet2=sheets2[0]                            #获取第一张表单

    # 获取行数和列数
    nrows = sheet1.nrows
    ncols = sheet1.ncols
    #print(nrows,ncols)                      #185,16

    # #开始计算每一列的平均值
    # #print(sheet1.col_values(1))
    # average=[0]                            #用于记录每一列的平均值,第一列是名字，用0占位
    # for i in range(1,ncols):                  #第一列是序号不读进去
    #     col_i=sheet1.col_values(i)
    #     col_i.pop(0)                          #把每一列的第一个中文标题去掉
    #     average.append(sum(col_i)/(nrows-1))
    # #print(average)

    nums=[]          #记录被挑选的图片的序号
    #print(nums)
    #开始遍历每一张图的每一个指标，来筛选全部指标都大于均值的图片
    for i in range(1,nrows):                #第一行不要,因是抬头，后面每一行就是一张图
        c=0                                 #用于记录一张图片有几个指标大于平均值(最后一个指标是小于平均值才更好)
        row1_i=sheet1.row_values(i)
        row2_i = sheet2.row_values(i)
        for j in metric_nums:
            if(j!=ncols-1):
                if (row1_i[j] >= row2_i[j]):
                    c += 1
            else:
                if (row1_i[j] <= row2_i[j]):
                    c += 1

        if(c>=k):                            #如果满足筛选条件，就选
            nums.append(i)
    #打印指定的目标指标名字
    metric_name = sheet1.row_values(0)
    ret=""
    for i in metric_nums:
        ret+=metric_name[i]+" "
    print(ret)
    #打印选出来的图片数量和编号
    print(len(nums))
    print(nums)
    #将选出来的图片的名字写入txt文件
    with open(txt_name,'w') as f:
        for n in nums:
            f.write(str(n) + '.jpg' + '\n')



if(__name__=='__main__'):
    select(excel_path1,excel_path2,metric_nums,txt_name,k)