.PHONY: test
test:
	DEBUG=1 \
	julia --color=yes test/runtests.jl

# Compare the generated code between the first and last debug dumps
FIRST=$(shell echo /tmp/JuliaCUDA_*/ | xargs -n 1 echo | sort -t _ -k 2,2 -n | head -n 1)
LAST=$(shell echo /tmp/JuliaCUDA_*/ | xargs -n 1 echo | sort -t _ -k 2,2 -n | tail -n 1)
.PHONY: diff
diff:
	-$(MAKE) test
	diff -Nur "$(FIRST)" "$(LAST)" && echo "No differences!"
