CC?=gcc
CXX?=g++


ifdef SLEEF_DIR
INCLUDE+=-I$(SLEEF_DIR)/include
LIB+=-L$(SLEEF_DIR)/lib
endif
WARNINGS+=-Wall -Wextra -Wpointer-arith -Wformat -Wunused-variable -Wno-attributes -Wno-ignored-qualifiers -Wno-unused-function -Wdeprecated -Wno-deprecated-declarations \
    -Wno-deprecated-copy # Because of Boost.Fusion

all: libkl.so llr_testing

libkl.o: libkl.c libkl.h
	$(CC) -fPIC -O3 -march=native $< -o $@ -c $(INCLUDE) $(LIB) $(WARNINGS)

libkl.so: libkl.o
	$(CC) -shared $<  -o $@ $(INCLUDE) $(LIB) -lsleef $(WARNINGS)

%: %.cpp libkl.so
	$(CXX) -L. -lkl $< -o $@ -O3 -Wall -Wextra $(WARNINGS)
