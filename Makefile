.PHONY: clean

all: index.html

index.html: index.bs
	python3 tools/reformat-js.py
ifdef online
	curl https://www.w3.org/publications/spec-generator/ \
		-F file=@index.bs -F type=bikeshed-spec -F die-on=warning > index.html
else
	bikeshed --die-on=fatal spec index.bs
endif
	node tools/lint.mjs --verbose

clean:
	rm -f index.html
