all: main gen

main: main.o rbm.o
	g++ -o main main.o rbm.o

main.o: main.cpp rbm.h
	g++ -c main.cpp

gen: gen.o rbm.o
	g++ -o gen gen.o rbm.o

gen.o: gen.cpp rbm.h
	g++ -c gen.cpp

rbm.o: rbm.cpp rbm.h
	g++ -c rbm.cpp

clean:
	rm main.o main gen.o gen rbm.o

dataclean:
	rm ./data/image-*.dat

