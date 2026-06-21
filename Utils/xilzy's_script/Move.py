# ### 根据txt把指定的图片移到指定的位置
# import shutil
# ##参数
# # source_path="F:\论文整理构思\评价指标\傅羽\\test_imgs\mri-ct"
# # target_path="E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-ct"
# # txt_path=target_path+"\mri-ct_selected_picture_4.txt"
#
# # source_path="F:\论文整理构思\评价指标\傅羽\\test_imgs\mri-pet"
# # target_path="E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-pet"
# # txt_path=target_path+"\mri-pet_selected_picture_2.txt"
#
# source_path="F:\论文整理构思\评价指标\傅羽\\test_imgs\mri-spect"
# target_path="E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-spect"
# txt_path=target_path+"\mri-spect_selected_picture_2.txt"
#
# def move(source_path,target_path,txt_path):
#     #首先从大路径拼接子路径
#     a=source_path.split('\\')
#     a=a[-1].split('-')
#     source_path1=  source_path + '\\' + a[0]+ '\\'
#     source_path2 = source_path + '\\' + a[1]+ '\\'
#     target_path1=  target_path + '\\' + a[0]
#     target_path2 = target_path + '\\' + a[1]
#     #print(source_path1,target_path1)
#
#     #然后从txt中读取要复制的图片文件名
#     with open(txt_path,'r') as f:
#         files=f.readlines()
#     for file in files:
#         file_name = file[:-1]          #去掉末尾的换行符
#         shutil.copy(source_path1 + file_name, target_path1)
#         shutil.copy(source_path2 + file_name, target_path2)
#
#
# if(__name__=="__main__"):
#     move(source_path,target_path,txt_path)

### 根据txt把指定的图片移到指定的位置
import shutil
##参数
source_path="E:\论文写作\杨老师\论文投稿\第二篇\材料\检验图片\gfp-pci_all"
target_path="E:\论文写作\杨老师\论文投稿\第二篇\材料\检验图片\gfp-pci"
txt_path=target_path+"\第三篇_selected_picture_8.txt"

def move(source_path,target_path,txt_path):
    #首先从大路径拼接子路径
    a=target_path.split('\\')
    a=a[-1].split('-')
    source_path1=  source_path + '\\' + a[0]+ '\\'
    source_path2 = source_path + '\\' + a[1]+ '\\'
    target_path1=  target_path + '\\' + a[0]
    target_path2 = target_path + '\\' + a[1]
    print(source_path1,target_path1,source_path2,target_path2)

    #然后从txt中读取要复制的图片文件名
    with open(txt_path,'r') as f:
        files=f.readlines()
    for file in files:
        file_name = file[:-1]          #去掉末尾的换行符
        shutil.copy(source_path1 + file_name, target_path1)
        shutil.copy(source_path2 + file_name, target_path2)


if(__name__=="__main__"):
    move(source_path,target_path,txt_path)