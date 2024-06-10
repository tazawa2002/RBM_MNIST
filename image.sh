#!/bin/bash

# 入力ファイル名
input_file="./data/data.dat"
# 出力ファイル名
output_file="./data/image.dat"

# 入力ファイルの内容を読み込む
data=$(awk 'NR=50' $input_file)

# 出力ファイルを初期化
> $output_file

# データを配列に分割
data_array=($data)

# 28x28の行列に変換して出力ファイルに書き込む
for ((i=0; i<28; i++)); do
  start=$((i * 28))
  end=$((start + 28))
  line=(${data_array[@]:$start:28})
  echo "${line[@]}" >> $output_file
done

echo "Conversion completed. Output saved to $output_file"

gnuplot ./plt/image.plt