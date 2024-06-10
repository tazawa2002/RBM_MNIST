set terminal gif animate delay 1 font "VL PGothic,10" size 720, 720
set output "./graph/image.gif"

unset key

# ファイル数を数える
file_count = system("ls ./data/image-*.dat | wc -l")
file_count = int(file_count)

set xrange [-0.5:27.5]
set yrange [27.5:-0.5]
set size ratio -1
unset key
unset colorbox
set palette defined ( 0 'white', 1 'black' )

do for [i=0:file_count-1]{
    plot sprintf("./data/image-%d.dat", i) matrix with image
}