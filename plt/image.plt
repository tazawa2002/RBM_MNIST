set terminal pngcairo size 560, 560
set output 'image.png'
set xrange [-0.5:27.5]
set yrange [27.5:-0.5]
set size ratio -1
unset key
unset colorbox
set palette defined ( 0 'white', 1 'black' )

plot "./data/image.dat" matrix with image


