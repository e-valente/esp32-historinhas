// Portable compact UTF-8 ByteLevel-BPE encoder for Historinhas. The same code
// runs on the host and ESP32 and reads the BTK1 asset produced by
// firmware/historinhas/tools/tokenizer_asset.py.
//
// Historinhas is trained with ByteLevel(use_regex=false), therefore the whole
// valid UTF-8 prompt is one pre-token. BTK1 stores dense token ids and ranked
// BPE merges in little-endian form.

#ifndef BPE_TOKENIZER_H
#define BPE_TOKENIZER_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define BTK_FORMAT_VERSION 2
#define BTK_HEADER_BYTES 52
#define BTK_BYTE_TABLE_BYTES 512
#define BTK_MERGE_ENTRY_BYTES 6
#define BTK_MAX_INPUT_BYTES 512

/* Token ids and merge ranks are uint16 in the asset, so neither count can
 * exceed this. A corrupt header would otherwise send the binary search walking
 * past the end of the image. */
#define BTK_MAX_TABLE_ENTRIES 0x10000

/* Encoder failures. Negative so a caller can test for < 0 in bulk. */
#define BTK_ERR_TOO_LONG (-2)
#define BTK_ERR_CAPACITY (-4)
#define BTK_ERR_INVALID_UTF8 (-5)

typedef struct {
  const uint8_t *asset;
  uint32_t active_vocab;
  uint32_t merge_count;
  /* Token id of merge rank 0, so rank r yields merge_base + r. Stored rather
   * than assumed, because it follows from where the vocabulary happens to place
   * its 256 single-byte symbols. */
  uint32_t merge_base;
  const uint8_t *byte_ids;
  const uint8_t *merges;
} BpeTokenizer;

static uint16_t btk_u16(const uint8_t *p) {
  return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t btk_u32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
         ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

// Returns 0 on success, -1 on an asset this encoder cannot read.
//
// asset_size is required, not advisory: the merge count is a field of the asset
// itself, so without a length there is nothing to check it against and a
// truncated table sends btk_merge_rank reading past the end of the image.
// The generated header supplies TOKENIZER_ENCODER_ASSET_SIZE for this argument.
//
// Nothing is written to tok until every check has passed, so a rejected asset
// cannot leave a half-populated tokenizer behind.
static int bpe_tokenizer_load(
    const uint8_t *asset, size_t asset_size, BpeTokenizer *tok) {
  if (asset == NULL || tok == NULL) return -1;
  /* Enough bytes to read the header and byte table before trusting any field. */
  if (asset_size < BTK_HEADER_BYTES + BTK_BYTE_TABLE_BYTES) return -1;
  if (memcmp(asset, "BTK1", 4) != 0) return -1;
  if (btk_u32(asset + 4) != BTK_FORMAT_VERSION) return -1;

  uint32_t active_vocab = btk_u32(asset + 8);
  uint32_t merge_count = btk_u32(asset + 12);
  /* asset + 16 is reserved and deliberately not read. */
  uint32_t merge_base = btk_u32(asset + 20);

  /* The 256 single-byte symbols are always present, so the vocabulary cannot be
   * smaller than that. The upper bound is what a uint16 token id can address. */
  if (active_vocab < 256 || active_vocab > BTK_MAX_TABLE_ENTRIES) return -1;
  if (merge_count == 0 || merge_count > BTK_MAX_TABLE_ENTRIES) return -1;
  /* Rank 0 must name a real token, and rank merge_count-1 must too. Both bounds
   * matter: btk_encode_piece emits merge_base + rank as a uint16, so an id at or
   * past active_vocab is either outside the vocabulary or a wrapped value.
   * active_vocab is already capped above, so this also keeps the largest emitted
   * id, merge_base + merge_count - 1, inside uint16. */
  if (merge_base == 0 || merge_base >= active_vocab) return -1;
  if (merge_base + merge_count > active_vocab) return -1;

  /* Only now is merge_count trustworthy enough to size the table with. */
  size_t required = (size_t)BTK_HEADER_BYTES + BTK_BYTE_TABLE_BYTES +
                    (size_t)merge_count * BTK_MERGE_ENTRY_BYTES;
  if (asset_size < required) return -1;

  tok->asset = asset;
  tok->active_vocab = active_vocab;
  tok->merge_count = merge_count;
  tok->merge_base = merge_base;
  tok->byte_ids = asset + BTK_HEADER_BYTES;
  tok->merges = tok->byte_ids + BTK_BYTE_TABLE_BYTES;
  return 0;
}

static uint16_t btk_byte_id(const BpeTokenizer *tok, uint8_t byte) {
  return btk_u16(tok->byte_ids + (size_t)byte * 2);
}

// Rank of the pair, or -1 if it is not a merge. Binary search: the table is
// sorted by key, and at 7000-odd entries a linear scan per adjacent pair per
// round would dominate encoding.
static int btk_merge_rank(
    const BpeTokenizer *tok, uint16_t left, uint16_t right) {
  uint32_t key = ((uint32_t)left << 16) | right;
  uint32_t lo = 0, hi = tok->merge_count;
  while (lo < hi) {
    uint32_t mid = lo + (hi - lo) / 2;
    const uint8_t *entry = tok->merges + (size_t)mid * BTK_MERGE_ENTRY_BYTES;
    uint32_t candidate = btk_u32(entry);
    if (candidate < key) lo = mid + 1;
    else hi = mid;
  }
  if (lo == tok->merge_count) return -1;
  const uint8_t *entry = tok->merges + (size_t)lo * BTK_MERGE_ENTRY_BYTES;
  return btk_u32(entry) == key ? (int)btk_u16(entry + 4) : -1;
}

// Ranked BPE over one pre-token: repeatedly apply the lowest-ranked adjacent
// merge until none applies.
static int btk_encode_piece(
    const BpeTokenizer *tok,
    const uint8_t *piece,
    int piece_len,
    uint16_t *out,
    int out_cap) {
  if (piece_len > BTK_MAX_INPUT_BYTES || piece_len > out_cap)
    return BTK_ERR_TOO_LONG;
  int n = piece_len;
  for (int i = 0; i < n; i++) out[i] = btk_byte_id(tok, piece[i]);

  for (;;) {
    int best_rank = -1;
    uint16_t best_left = 0, best_right = 0;
    for (int i = 0; i + 1 < n; i++) {
      int rank = btk_merge_rank(tok, out[i], out[i + 1]);
      if (rank >= 0 && (best_rank < 0 || rank < best_rank)) {
        best_rank = rank;
        best_left = out[i];
        best_right = out[i + 1];
      }
    }
    if (best_rank < 0) break;

    int write = 0;
    for (int read = 0; read < n;) {
      if (read + 1 < n && out[read] == best_left &&
          out[read + 1] == best_right) {
        out[write++] = (uint16_t)(tok->merge_base + best_rank);
        read += 2;
      } else {
        out[write++] = out[read++];
      }
    }
    n = write;
  }
  return n;
}

// Validate a non-NUL UTF-8 string before presenting its raw bytes to a
// ByteLevel tokenizer. ByteLevel itself operates on bytes, but validation keeps
// malformed serial input from being accepted as text that Python could not have
// received from a normal UTF-8 terminal.
static int btk_valid_utf8(const uint8_t *s, int len) {
  for (int i = 0; i < len;) {
    uint8_t c = s[i++];
    if (c < 0x80) continue;
    int more;
    uint32_t value;
    if (c >= 0xc2 && c <= 0xdf) { more = 1; value = c & 0x1f; }
    else if (c >= 0xe0 && c <= 0xef) { more = 2; value = c & 0x0f; }
    else if (c >= 0xf0 && c <= 0xf4) { more = 3; value = c & 0x07; }
    else return 0;
    if (i + more > len) return 0;
    for (int j = 0; j < more; j++) {
      uint8_t tail = s[i++];
      if ((tail & 0xc0) != 0x80) return 0;
      value = (value << 6) | (tail & 0x3f);
    }
    // Reject overlong encodings, surrogate code points, and code points beyond
    // Unicode's range. These are all byte sequences Python's UTF-8 decoder
    // rejects too.
    if ((more == 1 && value < 0x80) || (more == 2 && value < 0x800) ||
        (more == 3 && value < 0x10000) || (value >= 0xd800 && value <= 0xdfff) ||
        value > 0x10ffff) return 0;
  }
  return 1;
}

// Encoder for Historinhas' ByteLevel BPE contract. There is exactly one
// pre-token, so ranked BPE is applied to the complete UTF-8 byte string.
static int bpe_encode_utf8(
    const BpeTokenizer *tok,
    const char *text,
    uint16_t *out,
    int out_cap) {
  int len = (int)strlen(text);
  if (len > BTK_MAX_INPUT_BYTES) return BTK_ERR_TOO_LONG;
  if (!btk_valid_utf8((const uint8_t *)text, len)) return BTK_ERR_INVALID_UTF8;
  return btk_encode_piece(tok, (const uint8_t *)text, len, out, out_cap);
}

#endif
