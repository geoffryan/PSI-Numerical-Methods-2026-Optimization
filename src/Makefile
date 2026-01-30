APP = time_test
LIB = libmycalc.so

CC = clang-mp-18

CFLAGS = -O3 -Wall -fopenmp -arch arm64
LDFLAGS = -lm

default: $(APP)

.PHONY: clean

$(LIB): mycalc.c mycalc.h
	$(CC) -o $@ $< $(CFLAGS) -fPIC -shared
	
$(APP): test_main.c $(LIB)
	$(CC) -o $@ -L. -lmycalc $< $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(LIB) $(APP)
