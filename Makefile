.PHONY=all clean

CC?=gcc
CXX?=g++


ifdef SLEEF_DIR
INCLUDE+=-I$(SLEEF_DIR)/include
LIB+=-L$(SLEEF_DIR)/lib -lsleef
endif
WARNINGS+=-Wall -Wextra -Wpointer-arith -Wformat -Wunused-variable -Wno-attributes -Wno-ignored-qualifiers -Wno-unused-function -Wdeprecated -Wno-deprecated-declarations

EXTRA+=-fPIC -O3 -march=native -flto
ND=-DNDEBUG

all: libkl.so libkl.a libkl.dylib libkl.o

libkl.o: libkl.c libkl.h
	$(CC) $< -o $@ -c $(INCLUDE) $(LIB) $(WARNINGS) $(EXTRA) -std=c11 $(ND)

libkl.so: libkl.c
	$(CC) -shared $<  -o $(shell pwd)/$@ $(INCLUDE) $(LIB) -lsleef $(WARNINGS) $(EXTRA) -std=c11 $(ND)

libkl.dylib: libkl.c
	$(CC) -dynamiclib $<  -o $(shell pwd)/$@ $(INCLUDE) $(LIB) -lsleef $(WARNINGS) $(EXTRA) -std=c11 $(ND)

libkl.a: libkl.o
	$(AR) rcs $@ $<

%: %.cpp libkl.so
	$(CXX) -L. -lkl $< -o $@ -Wall -Wextra $(WARNINGS) $(EXTRA) $(INCLUDE) $(LIB) -std=c++11

testho: testho.cpp
	$(CXX) -L. -lkl $< -o $@ -Wall -Wextra $(WARNINGS) $(EXTRA) $(INCLUDE) $(LIB) -std=c++11

clean:
	rm -f libkl.so llr_testing libkl.a libkl.o test llr_testing test libkl.dylib
