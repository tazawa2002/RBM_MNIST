set terminal gif animate delay 1 font "VL PGothic,10" size 720, 720
unset key

number = 1

set output sprintf("./graph/image%d.gif", number)

# ファイル数を数える
file_count = system(sprintf("ls ./data/image%d-*.dat | wc -l", number))
file_count = int(file_count)

set xrange [-0.5:27.5]
set yrange [27.5:-0.5]
set size ratio -1
unset key
unset colorbox
set palette defined ( 0 'white', 1 'black' )
# set palette defined ( 0 'black', 1 'white' )

do for [i=0:file_count-1]{
    plot sprintf("./data/image%d-%d.dat", number, i) matrix with image
}