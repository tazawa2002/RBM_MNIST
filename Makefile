CXXFLAGS = -std=c++11

all: main gen

main: main.o rbm.o
	g++ -o main main.o rbm.o $(CXXFLAGS)

main.o: main.cpp rbm.h
	g++ -c main.cpp $(CXXFLAGS)

gen: gen.o rbm.o
	g++ -o gen gen.o rbm.o $(CXXFLAGS)

gen.o: gen.cpp rbm.h
	g++ -c gen.cpp $(CXXFLAGS)

rbm.o: rbm.cpp rbm.h
	g++ -c rbm.cpp $(CXXFLAGS)

clean:
	rm main.o main gen.o gen rbm.o

dataclean:
	rm ./data/image-*.dat

