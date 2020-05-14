all: index.html

index.html: index.bs
ifdef online
	curl https://api.csswg.org/bikeshed/ -F file=@index.bs -F output=err | tee /dev/stderr | grep -e ".*" > /dev/null; \
	if [ $$? -eq 0 ]; \
		then exit 1; \
	fi; \
	curl https://api.csswg.org/bikeshed/ -F file=@index.bs -F force=1 > index.html
else
	bikeshed --die-on=everything spec index.bs
endif