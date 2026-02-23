"""
test_image_vision.py
--------------------
Quick sanity check that the LLM can see and describe images.

Usage:
    python test_image_vision.py path/to/image.png
    python test_image_vision.py path/to/image.png --provider anthropic
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from FunSearch.llm_caller import LLMCaller


def test_image_vision(image_path: Path, provider: str = "google") -> None:
    print(f"Provider : {provider}")
    print(f"Image    : {image_path}")
    print(f"Exists   : {image_path.exists()}")
    print()

    if not image_path.exists():
        print(f"ERROR: image not found at {image_path}")
        return

    caller = LLMCaller(provider=provider, temperature=0.5)

    segments = [
        ("""You are an AI scientist doing symbolic regression.
Pay close attention to the attached image. It shows some data (red scatterpoints) that we need to model the relationship for (gray line).

Please write two python functions which will be used to fit the data. Make sure to import all the packages you will be using, such as import numpy as np 

def model(x, params):
 
def estimate_params(x,y):""", [image_path]),
    ]

    print("Sending image to LLM...")
    response = caller.call_interleaved_sync(segments)

    print("\n--- Response ---")
    print(response if response else "No response received.")


if __name__ == "__main__":
    test_img_path = Path('./problems/sinewave/funsearch/seed/program_1/evaluation_figure.png')
    test_image_vision(test_img_path, provider="google")