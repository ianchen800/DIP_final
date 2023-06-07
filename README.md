# DIP_final

members: 丁羿涵 施宇飛 陳羿穎


To test the evaluation of our SLIC on BSD dataset:
1. Download BSD 300 and put the "train" folder and "test" folder under ./data/berkely_dataset/images
2. Use the command ```bash ./evaluation/eval.sh```
* or use command ```python ./evaluation/main.py  {dataset name} {data_dir} {segmentation file dir} {k} {m}``` for customized setting.

To get the single image result of SLIC:
1. use the command ```python ./slic.py {input image} {output image} {k} {m}```
