from PIL import Image
import glob
import os

"""
サンプル画像を読み込んでテストデータを作成
* オリジナルサイズ:64x64
* 出力サイズ：64/n x 64/n
"""

input_size = 64
scale = 2

input_dir = ".\\dataset\\fsrcnn_input\\"
output_dir = ".\\dataset\\fsrcnn_output\\"

if not os.path.exists(input_dir):
    os.mkdir(input_dir)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# ディレクトリ配下の画像を読み込む
for i, f in enumerate(glob.glob('.\dataset\org\*.bmp')):
    img = Image.open(f)
    # 低分解能画像は画像サイズを小さくする
    org_x, org_y = img.size[0], img.size[1]
    x = int(round(float(org_x/scale)))
    y = int(round(float(org_y/scale)))
    low_res_img = img.resize((x, y), Image.ANTIALIAS)
    # ハイレゾ画像はそのまま
    high_res_img = img

    # 出力
    counter = "{0:05d}".format(i)
    low_res_img.save(input_dir + counter + ".bmp")
    high_res_img.save(output_dir + counter + ".bmp")
