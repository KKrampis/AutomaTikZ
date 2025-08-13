import os
from automatikz.infer import TikzGenerator, load

# Read HF API token from environment
token = os.getenv("HF_TOKEN")
if not token:
    raise RuntimeError("Please set the environment variable HF_TOKEN with your Hugging Face API token.")

# Read inference endpoint (or adapter repo) from environment or use default public TikZ-CLiMA model
endpoint = os.getenv(
    "HF_INFERENCE_ENDPOINT",
    "https://api-inference.huggingface.co/models/nllg/tikz-clima-13b",
)

generate = TikzGenerator(
    *load(endpoint, token=token),
    stream=False,
)
caption = (
    "Generate TikZ graphics tex code for a visual representation of a multi-layer perceptron: an interconnected network of nodes, showcasing the structure of input, hidden, and output layers that facilitate complex pattern recognition."
)

tikzdoc = generate(caption) # streams generated tokens to stdout
tikzdoc.save("mlp.tex") # save the generated code
if tikzdoc.has_content: # true if generated tikzcode compiles to non-empty pdf
    tikzdoc.rasterize().show() # raterize pdf to a PIL.Image and show it
    tikzdoc.save("mlp.pdf") # save the generated pdf
