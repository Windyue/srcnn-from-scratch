from PIL import Image, ImageFilter
import glob
import os

"""
サンプル画像を読み込んでテストデータを作成
* オリジナルサイズ:64x64
* 出力サイズ：52x52
"""

input_size = 64
output_size = 64 - (9 - 1) - (1 - 1) - (5 - 1)

x = (input_size - output_size) / 2

input_dir = ".\\dataset\\input\\"
output_dir = ".\\dataset\\output\\"

if not os.path.exists(input_dir):
    os.mkdir(input_dir)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# ディレクトリ配下の画像を読み込む
for i, f in enumerate(glob.glob('.\dataset\org\*.bmp')):
    img = Image.open(f)
    # ぼかし処理
    blur_img = img.filter(ImageFilter.BLUR)
    # ハイレゾ画像をくりぬく
    high_res_img = img.crop((x, x, x + output_size, x + output_size))

    # 出力
    counter = "{0:05d}".format(i)
    blur_img.save(input_dir + counter + ".bmp")
    high_res_img.save(output_dir + counter + ".bmp")
