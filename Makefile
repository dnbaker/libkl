.PHONY=all clean

CC?=gcc
CXX?=g++


ifdef SLEEF_DIR
INCLUDE+=-I$(SLEEF_DIR)/include
LIB+=-L$(SLEEF_DIR)/lib #-lsleef
endif
WARNINGS+=-Wall -Wextra -Wpointer-arith -Wformat -Wunused-variable -Wno-attributes -Wno-ignored-qualifiers -Wno-unused-function -Wdeprecated -Wno-deprecated-declarations

BASEFLAGS= -fPIC -O3 -march=native
EXTRA+=
ND=-DNDEBUG

all: libkl.so libkl.a libkl.o
KLCMD=
ifeq ($(shell uname),Darwin)
    KLCMD+= && $(CC) -dynamiclib libkl.c  -o $(shell pwd)/libkl.dylib $(INCLUDE) $(LIB) $(WARNINGS) $(EXTRA) -std=c11 $(ND)
endif

libkl.o: libkl.c libkl.h
	$(CC) $< -o $@ -c $(INCLUDE) $(LIB) $(WARNINGS) $(EXTRA) -std=c11 $(ND) $(BASEFLAGS)

libkl.so: libkl.c
	$(CC) -shared $<  -o $(shell pwd)/$@ $(INCLUDE) $(LIB) $(WARNINGS) $(EXTRA) -std=c11 $(ND) $(KLCMD) $(BASEFLAGS)

libkl.a: libkl.o
	$(AR) rcs $@ $<

%: %.cpp libkl.so
	$(CXX) -L. -lkl $< -o $@ -Wall -Wextra $(WARNINGS) $(EXTRA) $(INCLUDE) $(LIB) -std=c++11 $(BASEFLAGS)

testho: testho.cpp
	$(CXX) -L. -lkl $< -o $@ -Wall -Wextra $(WARNINGS) $(EXTRA) $(INCLUDE) $(LIB) -std=c++11

clean:
	rm -f libkl.so llr_testing libkl.a libkl.o test llr_testing test libkl.dylib
