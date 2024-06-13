CXXFLAGS = -std=c++11

all: main gen

main: main.o rbm.o rbm_mnist.o
	g++ -o main main.o rbm.o rbm_mnist.o $(CXXFLAGS)

main.o: main.cpp rbm.h rbm_mnist.h
	g++ -c main.cpp $(CXXFLAGS)

gen: gen.o rbm.o rbm_mnist.o
	g++ -o gen gen.o rbm.o rbm_mnist.o $(CXXFLAGS)

gen.o: gen.cpp rbm.h rbm_mnist.h
	g++ -c gen.cpp $(CXXFLAGS)

rbm.o: rbm.cpp rbm.h
	g++ -c rbm.cpp $(CXXFLAGS)

rbm_mnist.o: rbm_mnist.cpp rbm_mnist.h rbm.h
	g++ -c rbm_mnist.cpp $(CXXFLAGS)

clean:
	rm -f main.o main gen.o gen rbm.o rbm_mnist.o

dataclean:
	rm -f ./data/image-*.dat
