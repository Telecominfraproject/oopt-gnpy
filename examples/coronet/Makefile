REGIONS = europe asia conus
TARGETS = $(foreach region,$(REGIONS),coronet.$(region).json)

all: $(TARGETS)

$(TARGETS): convert.py CORONET_Global_Topology.xls
	python $< -f $(subst .json,,$(subst coronet.,,$@)) > $@

.PHONY: clean
clean:
	-rm $(TARGETS) -f
