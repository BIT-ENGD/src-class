.PHONY: all clean

all: fatigue.svg

clean:
	rm fatigue.svg

fatigue.svg: fatigue.dot
	dot -Tsvg $< > $@
