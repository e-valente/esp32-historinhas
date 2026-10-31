// PLE Historinhas inference on the ESP32-S3.
//
// The 28.9M-param model (14.9MB, 4-bit) lives in a flash 'model' partition,
// memory-mapped. Placement follows reads-per-token rather than what happens to
// fit:
//
//   flash   PLE table + token embedding   one row of the 25.2M-parameter
//                                         table per token
//   PSRAM   staged int8 core + head, KV   read once per position
//   SRAM    scratch + norm vectors        touched many times per token
//
// The logits array stays in PSRAM and is sized from the loaded artifact, so the
// active Historinhas vocabulary rather than a stale compiled-in number decides
// its footprint.
//
// Same llm.h that is verified against PyTorch on the host; only the platform
// hooks differ here.

#include "esp_partition.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"

// int8 activations, required by the staged int8 kernel. Not bit-exact against
// the fp32 golden; verify.c must be built without this flag. Measure the small
// additional loss with runtime/host_verify/ppl.c before shipping an artifact.
#define LLM_INT8_ACT 1
#define LLM_PROFILE 1
#define LLM_PROFILE_NOW() esp_timer_get_time()
#include "../../runtime/llm.h"
#include "../../runtime/bpe_tokenizer.h"
#include "generated/vocab.h"
#include "generated/tokenizer_encoder.h"
#include "story_sampler.h"

// Set to 1 once a display is wired up - see display.h.
// Leave 0 to run serial-only (no panel needed).
#define USE_DISPLAY 1
#if USE_DISPLAY
#include "display.h"
#endif

#define STORY_GENERATE_TOKENS 200
#define STORY_MIN_GENERATE_TOKENS 24
#define STORY_TOP_K 20
#define STORY_TEMPERATURE 0.60f
#define STORY_NO_REPEAT_NGRAM 4
#define STORY_SERIAL_RX_BYTES (2 * (BTK_MAX_INPUT_BYTES + 1))
static const char RESERVED_SPECIAL[] = "<|endoftext|>";

Model model;
Scratch s;
static BpeTokenizer tokenizer;
static bool ready = false;

// ---- allocation ------------------------------------------------------------
// Allocations are strict because memory placement is part of the runtime
// configuration. Allocation failure stops initialization.
static size_t psram_used = 0, sram_used = 0;

// The two LLM_Q8_MAX_INPUT int8 activation buffers (matvec_q8, matvec_par),
// which are static and so absent from the totals above. Together these report
// the managed hot set, not total SRAM usage; the free-SRAM figure printed at
// boot is the overall diagnostic.
#define STATIC_SRAM_BYTES (2 * LLM_Q8_MAX_INPUT)

static void *ps(size_t n) {
  void *p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM);
  if (p) psram_used += n;
  return p;
}
static void *ps_or_die(size_t n, const char *what) {
  void *p = ps(n);
  if (!p) {
    Serial.printf("FATAL: required PSRAM allocation failed: %s (%u bytes)\n",
                  what, (unsigned)n);
    while (1) delay(1000);
  }
  return p;
}
static void *sram_or_die(size_t n, const char *what) {
  void *p = heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!p) {
    Serial.printf("FATAL: required SRAM allocation failed: %s (%u bytes)\n",
                  what, (unsigned)n);
    while (1) delay(1000);
  }
  sram_used += n;
  return p;
}

// ---- dual-core int8 matvec -------------------------------------------------
// Serves both hooks. Below ~128 rows the task notify round trip costs more than
// the split saves, so small tensors run single-core.
static TaskHandle_t worker_h, main_h;
static const QT *job_t;
static const int8_t *job_xq;
static float job_xs;
static float *job_y;
static int job_split;

static void worker_main(void *) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    matvec_i8_range(job_t, job_xq, job_xs, job_y, 0, job_split);
    xTaskNotifyGive(main_h);
  }
}

static void matvec_par(const QT *t, const float *x, float *y) {
  static int8_t xq[LLM_Q8_MAX_INPUT];
  float xs;
  if (t->w8 == NULL || t->rows < 128) { MATVEC(t, x, y); return; }
  quantize_act(x, t->cols, xq, &xs);   // once; both cores read the result
  job_t = t; job_xq = xq; job_xs = xs; job_y = y; job_split = t->rows / 2;
  xTaskNotifyGive(worker_h);
  matvec_i8_range(t, xq, xs, y, job_split, t->rows);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
}

// Copy RMSNorm weights from mapped flash to internal SRAM.
static void copy_norms_to_sram() {
  Cfg *c = &model.c;
  int D = c->dim, L = c->n_layers, P = c->ple_dim;
  const float **vecs[3 * 32 + 2];
  int sizes[3 * 32 + 2], n_vec = 0;
  vecs[n_vec] = &model.ple_proj_norm; sizes[n_vec++] = P;
  for (int l = 0; l < L; l++) {
    vecs[n_vec] = &model.attn_norm[l]; sizes[n_vec++] = D;
    vecs[n_vec] = &model.ffn_norm[l];  sizes[n_vec++] = D;
    vecs[n_vec] = &model.ple_norm[l];  sizes[n_vec++] = D;
  }
  vecs[n_vec] = &model.out_norm; sizes[n_vec++] = D;
  for (int i = 0; i < n_vec; i++) {
    size_t bytes = (size_t)sizes[i] * sizeof(float);
    void *dst = sram_or_die(bytes, "norm vector");
    memcpy(dst, *vecs[i], bytes);
    *vecs[i] = (const float *)dst;
  }
  Serial.printf("norms  -> SRAM   %d vectors\n", n_vec);
}

static void alloc_scratch() {
  Cfg *c = &model.c;
  int D = c->dim, L = c->n_layers, P = c->ple_dim, F = c->ffn, S = c->seq_len;
  // hot working set -> internal SRAM
  s.x     = (float *)sram_or_die(D * 4, "x");
  s.h     = (float *)sram_or_die((F > D ? F : D) * 4, "h");
  s.qkv   = (float *)sram_or_die(3 * D * 4, "qkv");
  s.att   = (float *)sram_or_die(D * 4, "att");
  s.g1    = (float *)sram_or_die(F * 4, "g1");
  s.g2    = (float *)sram_or_die((P > F ? P : F) * 4, "g2");
  s.ple   = (float *)sram_or_die(L * P * 4, "ple");
  s.tmpP  = (float *)sram_or_die(L * P * 4, "tmpP");
  s.trow  = (float *)sram_or_die(L * P * 4, "trow");
  s.scores = (float *)sram_or_die(S * 4, "scores");
  // logits: out_vocab floats, 99 KiB here, read once per token. Left in PSRAM
  // rather than spend a fifth of internal SRAM on it.
  s.logits = (float *)ps_or_die((size_t)model.out_vocab * 4, "logits");
  // KV cache: 1.1MB, read once per position rather than per matvec.
  s.kcache = (float *)ps_or_die((size_t)L * S * D * 4, "kcache");
  s.vcache = (float *)ps_or_die((size_t)L * S * D * 4, "vcache");
}

static void blink(uint8_t g) {
#ifdef RGB_BUILTIN
  rgbLedWrite(RGB_BUILTIN, 0, g, g / 3);
#endif
}

// Emit one token to every active output (serial always; panel when enabled).
static void emit(int tok) {
  if (tok >= VOCAB_N) return;
  const unsigned char *bytes = VOCAB_BLOB + VOCAB_OFF[tok];
  int len = VOCAB_OFF[tok + 1] - VOCAB_OFF[tok];
  // Non-blocking: when no host is draining the USB-CDC buffer (running as a
  // standalone gadget on the display), skip the write instead of stalling the
  // whole generation once the TX buffer fills.
  if ((int)Serial.availableForWrite() >= len) Serial.write(bytes, len);
#if USE_DISPLAY
  display_puts(bytes, len);
#endif
}

// A new session always starts at position zero.  llm_forward overwrites the KV
// entries it can attend to, so no cache clear is needed between prompts.
static void generate(const char *prompt) {
  if (strstr(prompt, RESERVED_SPECIAL) != NULL) {
    Serial.printf("(reserved token %s is not allowed)\n", RESERVED_SPECIAL);
    return;
  }

  uint16_t ids[BTK_MAX_INPUT_BYTES];
  int n_prompt = bpe_encode_utf8(&tokenizer, prompt, ids,
                                           (int)(sizeof(ids) / sizeof(ids[0])));
  if (n_prompt == BTK_ERR_INVALID_UTF8) {
    Serial.println("(prompt must be valid UTF-8)");
    return;
  }
  if (n_prompt < 0) {
    Serial.println("(prompt cannot be tokenized)");
    return;
  }
  if (n_prompt == 0) {
    Serial.println("(empty prompt)");
    return;
  }
  if (n_prompt > model.c.seq_len - STORY_GENERATE_TOKENS) {
    Serial.printf("(prompt too long: at most %d tokens with %d generated)\n",
                   model.c.seq_len - STORY_GENERATE_TOKENS,
                   STORY_GENERATE_TOKENS);
    return;
  }

  Serial.print("A: ");
  Serial.print(prompt);
#if USE_DISPLAY
  // display_begin() leaves the optional panel unavailable when its I2C probe
  // fails. display_puts() handles that state, but display_home() dereferences
  // the panel object, so it must not run in serial-only mode.
  if (display_present) {
    display_home();
    display_puts((const unsigned char *)prompt, strlen(prompt));
  }
#endif

  int pos = 0, tok = 0;
  uint16_t generated[STORY_GENERATE_TOKENS];
  uint32_t rng = story_prompt_seed(prompt);
  for (int i = 0; i < n_prompt; i++) {
    llm_forward(&model, ids[i], pos++, &s);
    // A valid prompt can contain 56 tokens, several seconds of inference.
    // Yield during priming as well as decoding so the Arduino task WDT is fed.
    if ((i & 7) == 7) delay(0);
  }

  llm_profile_reset(&s);
  int64_t decode_us = 0;
  int64_t t_start = esp_timer_get_time();
  int decoded = 0;
  for (int step = 0; step < STORY_GENERATE_TOKENS && pos < model.c.seq_len;
       step++) {
    // EOT ends a training document. Suppress it only at the beginning so a
    // weak model cannot turn an interactive prompt into a one-word story.
    int forbidden = step < STORY_MIN_GENERATE_TOKENS ? VOCAB_EOT : -1;
    tok = story_sample_top_k(s.logits, model.out_vocab, STORY_TOP_K,
                             STORY_TEMPERATURE, &rng, generated, decoded,
                             STORY_NO_REPEAT_NGRAM, forbidden);
    if (tok < 0) {
      Serial.println("\n(sampling found no permitted token)");
      break;
    }
    if (tok == VOCAB_EOT) break;
    emit(tok);
    blink((step & 1) ? 40 : 8);

    int64_t d0 = esp_timer_get_time();
    llm_forward(&model, tok, pos++, &s);
    decode_us += esp_timer_get_time() - d0;
    generated[decoded] = (uint16_t)tok;
    decoded++;
    if ((step & 7) == 0) delay(0);  // feed the task WDT ~every 8 tokens
  }
  int64_t total_us = esp_timer_get_time() - t_start;

  Serial.printf("\n\n--- %d tokens in %.2f s ---\n", decoded, total_us / 1e6);
  if (decoded) {
    Serial.printf("throughput: %.2f tok/s   (%.1f ms/token)\n",
                  decoded * 1e6 / total_us, decode_us / 1000.0 / decoded);
  } else {
    Serial.println("throughput: n/a (no token emitted)");
  }
  if (s.profile.calls) {
    float count = (float)s.profile.calls * 1000.f;
    Serial.printf("profile ms/token: input %.1f | attn %.1f | ffn %.1f | ple %.1f | head %.1f\n",
                  s.profile.input_us / count, s.profile.attn_us / count,
                  s.profile.ffn_us / count, s.profile.ple_us / count,
                  s.profile.head_us / count);
  }
#if USE_DISPLAY
  if (display_present && decoded)
    display_stats(decoded * 1e6f / decode_us, decode_us / 1000.0f / decoded);
#endif
  blink(0);
}

void setup() {
  // The default USB-CDC RX queue is 256 bytes.  A complete accepted prompt is
  // 512 bytes plus its newline, and can arrive in one paste.
  size_t rx_bytes = Serial.setRxBufferSize(STORY_SERIAL_RX_BYTES);
  Serial.begin(115200);
  delay(1500);
  Serial.println("\n=== ESP32-S3 PLE Historinhas pt_BR ===");
  if (rx_bytes != STORY_SERIAL_RX_BYTES) {
    Serial.println("serial RX buffer allocation failed");
    return;
  }

  if (bpe_tokenizer_load(TOKENIZER_ENCODER_ASSET, TOKENIZER_ENCODER_ASSET_SIZE,
                         &tokenizer)) {
    Serial.println("tokenizer load failed");
    return;
  }

  const esp_partition_t *part = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "model");
  if (!part) { Serial.println("model partition not found"); return; }
  const void *base;
  esp_partition_mmap_handle_t h;
  esp_err_t err = esp_partition_mmap(part, 0, part->size,
                                     ESP_PARTITION_MMAP_DATA, &base, &h);
  if (err != ESP_OK) { Serial.printf("mmap failed: %d\n", err); return; }

  if (llm_load((const uint8_t *)base, &model)) { Serial.println("bad model magic"); return; }
  Cfg *c = &model.c;
  Serial.printf("model: Vin=%d Vout=%d D=%d L=%d H=%d F=%d P=%d  (mapped %.1f MB)\n",
                c->vocab, model.out_vocab, c->dim, c->n_layers, c->n_heads,
                c->ffn, c->ple_dim, part->size / 1e6);

#if USE_DISPLAY
  if (!display_begin()) Serial.println("no display answering; serial only");
#endif

  // The model header states how many logits it produces; vocab.h carries the
  // decode table. If they disagree, every emitted token would be decoded
  // against the wrong table.
  if (VOCAB_N != model.out_vocab) {
    Serial.printf("FATAL: tokenizer/model mismatch: vocab.h %d, model %d\n",
                  VOCAB_N, model.out_vocab);
    return;
  }
  if (VOCAB_EOT >= model.out_vocab) {
    Serial.printf("FATAL: tokenizer/model mismatch: EOT id %d, model %d\n",
                  VOCAB_EOT, model.out_vocab);
    return;
  }
  if ((int)tokenizer.active_vocab > c->vocab) {
    Serial.printf("FATAL: tokenizer/model mismatch: encoder emits up to %u, model reads %d\n",
                  (unsigned)tokenizer.active_vocab - 1, c->vocab);
    return;
  }
  if (c->seq_len < STORY_GENERATE_TOKENS) {
    Serial.printf("FATAL: model context %d is shorter than fixed generation %d\n",
                   c->seq_len, STORY_GENERATE_TOKENS);
    return;
  }

  alloc_scratch();
  copy_norms_to_sram();
  Serial.printf("hot set-> SRAM   %u B dynamic + %u B static = %u B managed\n",
                (unsigned)sram_used, (unsigned)STATIC_SRAM_BYTES,
                (unsigned)(sram_used + STATIC_SRAM_BYTES));

  // Stage every per-position tensor to int8 in PSRAM.
  int want = llm_core_stage_count(&model);
  int staged = llm_stage_core_int8_alloc(&model, ps);
  if (staged != want) {
    Serial.printf("FATAL: staged %d/%d core tensors\n", staged, want);
    while (1) delay(1000);
  }
  // The tied head is read per token, not per position, so the core helper does
  // not walk it. Stage it too: it is 85%% of the dense MACs.
  {
    void *b = ps_or_die(llm_stage_int8_bytes(&model.out_head), "staged head");
    llm_stage_int8(&model.out_head, b);
    ++staged;
  }
  Serial.printf("weights-> PSRAM  %d tensors int8, %.2f MB allocated\n",
                staged, psram_used / 1048576.0);

  main_h = xTaskGetCurrentTaskHandle();
  if (xTaskCreatePinnedToCore(worker_main, "mv", 4096, NULL, 2, &worker_h, 0) == pdPASS) {
    // After the worker exists: matvec_par notifies worker_h.
    model.layer_matvec = matvec_par;
    model.head_matvec  = matvec_par;
  } else {
    Serial.println("dual-core worker failed; running single core");
  }

  // FNV-1a over the mapped image. scripts/deploy.sh prints the same value for
  // the file it flashed; the two must agree.
  {
    const uint8_t *img = (const uint8_t *)base;
    uint32_t fp = 2166136261u;
    for (size_t i = 0; i < model.image_bytes; i++) { fp ^= img[i]; fp *= 16777619u; }
    Serial.printf("build: bytes=%u fp=%08x sram=%uB psram=%.2fMB\n",
                  (unsigned)model.image_bytes, fp,
                  (unsigned)(sram_used + STATIC_SRAM_BYTES),
                  psram_used / 1048576.0);
  }
  Serial.printf("free: sram %.0f KB | psram %.2f MB\n\n",
                heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024.0,
                heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1048576.0);

  ready = true;
  Serial.printf("sampling: top-k=%d temperature=%.2f no-repeat=%d\n",
                STORY_TOP_K, STORY_TEMPERATURE, STORY_NO_REPEAT_NGRAM);
  Serial.printf("READY> send a UTF-8 prompt; up to %d tokens will be generated\n",
                STORY_GENERATE_TOKENS);
}

void loop() {
  static char line[BTK_MAX_INPUT_BYTES + 1];
  static int len = 0;
  static bool overflowed = false;
  if (!ready) { delay(1000); return; }
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\r') continue;
    if (ch == '\n') {
      // Echo is handled by the device rather than assumed from the monitor:
      // pio device monitor and raw Linux cat sessions commonly disable local
      // echo. A newline still only submits the already collected line.
      Serial.println();
      line[len] = '\0';
      if (overflowed) Serial.println("(prompt too long: 512 bytes maximum)");
      else if (len) generate(line);
      else Serial.println("(empty prompt)");
      len = 0;
      overflowed = false;
      Serial.println("READY>");
    } else if (len < BTK_MAX_INPUT_BYTES) {
      line[len++] = ch;
      Serial.write((uint8_t)ch);
    } else {
      // Discard the rest through newline, so an oversized prompt cannot become
      // the start of the next session.
      overflowed = true;
      Serial.write((uint8_t)ch);
    }
  }
  delay(5);
}
