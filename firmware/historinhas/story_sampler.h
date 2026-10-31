// Deterministic sampling policy for Historinhas' interactive decoder.
//
// The model logits are still produced by llm.h. This header only selects the
// next token: top-k sampling avoids greedy repetition loops, while the n-gram
// gate prevents an already emitted phrase from being repeated verbatim.

#ifndef HISTORINHAS_STORY_SAMPLER_H
#define HISTORINHAS_STORY_SAMPLER_H

#include <math.h>
#include <stdint.h>

#define STORY_SAMPLER_MAX_TOP_K 64

typedef struct {
  int token;
  float logit;
} StoryCandidate;

// FNV-1a makes a request repeatable without carrying random state across
// sessions. A given prompt, artifact and firmware therefore has one output,
// which keeps device benchmarking meaningful.
static uint32_t story_prompt_seed(const char *text) {
  uint32_t h = 2166136261u;
  for (; *text; text++) {
    h ^= (uint8_t)*text;
    h *= 16777619u;
  }
  return h ? h : 1u;
}

static uint32_t story_prng_next(uint32_t *state) {
  uint32_t x = *state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x ? x : 1u;
  return *state;
}

// Would appending candidate repeat a completed n-gram in generated history?
// Prompt tokens are intentionally not included: a phrase present in the user
// prompt remains a valid continuation, but the model cannot loop on its own
// output. The normal setting is ngram=4.
static int story_repeats_ngram(const uint16_t *history, int history_count,
                               uint16_t candidate, int ngram) {
  if (ngram < 2 || history_count < ngram - 1) return 0;
  int prefix = ngram - 1;
  for (int start = 0; start + ngram <= history_count; start++) {
    int same = 1;
    for (int j = 0; j < prefix; j++) {
      if (history[start + j] != history[history_count - prefix + j]) {
        same = 0;
        break;
      }
    }
    if (same && history[start + prefix] == candidate) return 1;
  }
  return 0;
}

// Sample from the best top_k allowed logits. forbidden_token is normally EOT
// during the minimum-length period, otherwise -1. If every token is excluded
// by the n-gram gate (pathological tiny vocabulary), retry without that gate;
// EOT remains forbidden in both passes.
static int story_sample_top_k(const float *logits, int vocab, int top_k,
                              float temperature, uint32_t *rng,
                              const uint16_t *history, int history_count,
                              int no_repeat_ngram, int forbidden_token) {
  if (!logits || !rng || vocab <= 0 || top_k <= 0 || top_k > STORY_SAMPLER_MAX_TOP_K)
    return -1;

  StoryCandidate top[STORY_SAMPLER_MAX_TOP_K];
  int count = 0;
  for (int pass = 0; pass < 2 && count == 0; pass++) {
    int enforce_ngram = pass == 0;
    count = 0;
    for (int token = 0; token < vocab; token++) {
      if (token == forbidden_token) continue;
      if (enforce_ngram && story_repeats_ngram(history, history_count,
                                                (uint16_t)token, no_repeat_ngram))
        continue;
      float score = logits[token];
      if (count == top_k && score <= top[count - 1].logit) continue;
      int slot = count < top_k ? count++ : top_k - 1;
      while (slot > 0 && score > top[slot - 1].logit) {
        if (slot < top_k) top[slot] = top[slot - 1];
        slot--;
      }
      top[slot].token = token;
      top[slot].logit = score;
    }
  }
  if (count == 0) return -1;
  if (temperature <= 0.0f || count == 1) return top[0].token;

  // Subtracting the best logit keeps expf in a safe range. Top-k ranking is
  // temperature-independent; only the probability mass changes with it.
  float total = 0.0f;
  for (int i = 0; i < count; i++) {
    top[i].logit = expf((top[i].logit - top[0].logit) / temperature);
    total += top[i].logit;
  }
  float draw = ((float)story_prng_next(rng) * (1.0f / 4294967296.0f)) * total;
  for (int i = 0; i < count; i++) {
    draw -= top[i].logit;
    if (draw <= 0.0f) return top[i].token;
  }
  return top[count - 1].token;  // rounding at the upper edge of [0, total)
}

#endif
