CXX=icpx

CXXFLAGS=-O3 -march=native -DNDEBUG -qopenmp -qmkl=sequential
LDFLAGS=

default: task_lu.x

%.x: %.cpp
	$(CXX) -o $@ $< $(CXXFLAGS) ${LDFLAGS}

%.o: %.cpp
	$(CXX) -c -o $@ $< $(CXXFLAGS)

clean:
	-rm *.o *.s *.x
