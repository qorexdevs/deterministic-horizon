# --- Variables ---
VHS_IMG = ghcr.io/charmbracelet/vhs

# --- Targets ---
.PHONY: all tape-demo tape-img concat clean

all: tape-demo tape-img concat

# 1. Compile the first part of the tape (Python execution)
tape-demo:
	docker run --rm -u root -v "$(CURDIR)/../..:/vhs" --entrypoint /bin/sh $(VHS_IMG) -c \
	    "apt-get update && apt-get install -y git python3-pip && \
	    ln -sf /usr/bin/python3 /usr/bin/python && \
	    pip install numpy --break-system-packages && \
	    cd /vhs && \
	    pip install -e . --break-system-packages --ignore-requires-python && \
	    vhs assets/demo-gif-generator-scripts/code-run.tape"

# 2. Compile the second part of the tape (Chafa high-res image open)
tape-img:
	docker run --rm -u root -v "$(CURDIR)/../..:/vhs" --entrypoint /bin/sh $(VHS_IMG) -c \
	    "apt-get update && apt-get install -y chafa && \
	    cd /vhs && \
	    vhs assets/demo-gif-generator-scripts/open-image.tape"

# 3. Concatenate the two generated GIFs together with a unified color palette
concat:
	ffmpeg -y -i code-run.gif -i open-image.gif -filter_complex \
		"[0:v][1:v]concat=n=2:v=1:a=0,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
		../demo.gif

clean:
	rm -f demo.gif open-image.gif