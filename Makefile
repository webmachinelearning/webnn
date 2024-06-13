.PHONY: clean

all: index.html

index.html: index.bs
	python3 tools/reformat-js.py
ifdef online
	curl https://api.csswg.org/bikeshed/ -F file=@index.bs -F output=err | tee /dev/stderr | grep -e "ERROR" > /dev/null; \
	if [ $$? -eq 0 ]; \
		then exit 1; \
	fi; \
	curl https://api.csswg.org/bikeshed/ -F file=@index.bs -F force=1 > index.html
else
#	bikeshed -f spec index.bs
	bikeshed --die-on=fatal spec index.bs
endif
	node tools/lint.mjs --verbose

clean:
	rm -f index.html
