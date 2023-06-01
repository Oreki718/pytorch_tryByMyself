from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

img = Image.open("./imgs/sample.jpg")
print(img.size)
plt.imshow(img)
plt.show()

transformer = transforms.Compose([
    # 对载入的图片数据按照我们的需要进行缩放，传递给这个类的size可以是一个整型数据，也可以是一个类似于 (h ,w) 的序列。
    # 如果输入是个(h,w)的序列，h代表高度，w代表宽度，h和w都是int，则直接将输入图像resize到这个(h,w)尺寸，相当于force。
    # 如果使用的是一个整型数据，则将图像的短边resize到这个int数，长边则根据对应比例调整，图像的长宽比不变
    transforms.Resize(1000),

    # 以输入图的中心点为中心点为参考点，按我们需要的大小进行裁剪。
    # 传递给这个类的参数可以是一个整型数据，也可以是一个类似于(h,w)的序列。
    # 如果输入的是一个整型数据，那么裁剪的长和宽都是这个数值
    transforms.CenterCrop(1000),

    # 用于对载入的图片按我们需要的大小进行随机裁剪。
    # 传递给这个类的参数可以是一个整型数据，
    # 也可以是一个类似于(h,w)的序列。
    # 如果输入的是一个整型数据，那么裁剪的长和宽都是这个数值
    transforms.RandomCrop(1000),

    # 用于对载入的图片数据进行类型转换，
    # 将之前构成PIL图片的数据转换成Tensor数据类型的变量，
    # 让PyTorch能够对其进行计算和处理。
    transforms.ToTensor(),

    # 这里使用的是标准正态分布变换，
    # 这种方法需要使用原始数据的均值（Mean）和标准差（Standard Deviation）来进行数据的标准化，
    # 在经过标准化变换之后，数据全部符合均值为0、标准差为1的标准正态分布。
    # img should be Tensor Image
    transforms.Normalize((0, 0, 0), (0.8, 0.8, 0.8)),

    # 用于将Tensor变量的数据转换成PIL图片数据，
    # 主要是为了方便图片内容的显示。
    transforms.ToPILImage()
    # ......
])
img_c = transformer(img)
print(type(img_c))
# print(img_c.size())
# print(img_c.shape)
plt.imshow(img_c)
plt.show()