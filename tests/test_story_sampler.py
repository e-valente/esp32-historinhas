"""The ESP32 sampling policy is deterministic and breaks repeated n-grams."""

import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER_DIR = ROOT / "firmware/historinhas"

HARNESS = r'''
#include <assert.h>
#include <stdint.h>
#include "story_sampler.h"

int main(void) {
  uint16_t loop[] = {1, 2, 3, 4, 1, 2, 3};
  assert(story_repeats_ngram(loop, 7, 4, 4));
  assert(!story_repeats_ngram(loop, 7, 5, 4));
  assert(!story_repeats_ngram(loop, 2, 3, 4));

  float logits[] = {9.0f, 8.0f, 7.0f, 6.0f, 10.0f};
  uint32_t rng = 1;
  // Token 4 is the greedy choice, but it would complete 1,2,3,4 again.
  assert(story_sample_top_k(logits, 5, 1, 0.6f, &rng, loop, 7, 4, -1) == 0);
  // EOT (token zero here) can be suppressed independently of the loop gate.
  assert(story_sample_top_k(logits, 5, 1, 0.6f, &rng, loop, 0, 4, 0) == 4);

  uint32_t a = story_prompt_seed("Era uma vez");
  uint32_t b = story_prompt_seed("Era uma vez");
  assert(a == b && a != 0);
  int first = story_sample_top_k(logits, 5, 3, 0.6f, &a, loop, 0, 4, -1);
  int second = story_sample_top_k(logits, 5, 3, 0.6f, &b, loop, 0, 4, -1);
  assert(first == second);
  return 0;
}
'''


class TestStorySampler(unittest.TestCase):
    def test_sampling_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sampler.c"
            binary = Path(directory) / "sampler"
            source.write_text(HARNESS)
            subprocess.run(
                ["cc", "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror",
                 "-I", str(HEADER_DIR), "-o", str(binary), str(source), "-lm"],
                check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
