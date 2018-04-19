CXX = g++-5
CXXFLAGS = -std=c++14 -Wall -MMD
EXEC = myprogram
OBJECTS = main.o dataset.o regressor.o polynomial_regressor.o gaussian_process_regressor.o
DEPENDS = ${OBJECTS:.o=.d}

${EXEC}: ${OBJECTS}
	${CXX} ${CXXFLAGS} ${OBJECTS} -o ${EXEC}

-include ${DEPENDS}

.PHONY: clean

clean:
	rm ${OBJECTS} ${EXEC} ${DEPENDS}
