#将指定文件夹下的文件批量重命名，重命名格式为num.png(num连续编号)
import os
# 图片所在文件夹路径
#root_path = "F:\\论文整理构思\\构思实现\\EMFusion3\\CT-MRI\\test_imgs\\ct"
#root_path = "F:\\论文整理构思\\构思实现\\EMFusion3\\CT-MRI\\test_imgs\\mri"
#root_path = "F:\\论文整理构思\\构思实现\\EMFusion3\\PET-MRI\\test_imgs\\pet"
#root_path = "F:\\论文整理构思\\构思实现\\EMFusion3\\PET-MRI\\test_imgs\\mri"
#root_path = "F:\\论文整理构思\\构思实现\\EMFusion3\\SPECT-MRI\\test_imgs\\spect"
#root_path = "F:\\论文整理构思\\构思实现\\EMFusion3\\SPECT-MRI\\test_imgs\\mri"

# root_path = "E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-ct\ct"
#root_path = "E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-ct\mri"
# root_path = "E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-pet\pet"
# root_path = "E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-pet\mri"
# root_path = "E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-spect\spect"
# root_path = "E:\论文写作\杨老师\论文投稿\BSPC\第一篇文章\材料\检验图片\mri-spect\mri"

root_path = "E:\论文写作\杨老师\论文投稿\第二篇\材料\检验图片\gfp-pci\GFP"
# root_path = "E:\论文写作\杨老师\论文投稿\第二篇\材料\检验图片\gfp-pci\PCI"
filename_list = os.listdir(root_path)
print(filename_list)
for i in range(len(filename_list)):
    if filename_list[i].endswith('.jpg'):                  #这里限制图片后缀
        src_img_path = os.path.join(os.path.abspath(root_path), filename_list[i])
        new_img_path = os.path.join(os.path.abspath(root_path), str(i+1)+'.jpg') #这里限制图片后缀
        try:
            os.rename(src_img_path, new_img_path)
            print('converting %s to %s ...' % (src_img_path, new_img_path))
        except:
            continue