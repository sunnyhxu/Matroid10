#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#if __has_include(<nauty/nauty.h>)
#include <nauty/nauty.h>
#elif __has_include(<nauty.h>)
#include <nauty.h>
#else
#error "nauty headers not found"
#endif

namespace {

constexpr int kMaxN = 10;
constexpr uint64_t kFieldStride = 10007ULL;
constexpr uint64_t kSplitMixIncrement = 0x9e3779b97f4a7c15ULL;
constexpr int kExitUsageError = 2;
constexpr int kExitRuntimeError = 3;

enum class GenerationMode { kRepresentable, kSparsePaving };

struct GenConfig {
  GenerationMode mode = GenerationMode::kRepresentable;
  int field = 2;
  int rank_min = 4;
  int rank_max = 9;
  int n = 10;
  int threads = 1;
  uint64_t seed = 42;
  int max_seconds = 600;
  uint64_t max_trials = 0;
  uint64_t trial_start = 0;
  uint64_t trial_stride = 1;
  double sparse_accept_prob = 0.05;
  int sparse_min_ch = 1;
  int sparse_max_ch = 0;
  bool require_connected = true;
  std::string out_path = "artifacts/non_paving.jsonl";
  std::string stats_out_path;
  std::string config_path;
};

struct Stats {
  std::atomic<uint64_t> candidates{0};
  std::atomic<uint64_t> full_rank{0};
  std::atomic<uint64_t> non_paving{0};
  std::atomic<uint64_t> disconnected_filtered{0};
  std::atomic<uint64_t> unique_hits{0};
  std::atomic<uint64_t> duplicates{0};
  std::atomic<uint64_t> emitted_bases{0};
  std::atomic<uint64_t> sparse_rejected_min_ch{0};
  std::atomic<uint64_t> sparse_rejected_empty_bases{0};
  std::atomic<uint64_t> sparse_circuit_hyperplanes_total{0};
};

struct SharedState {
  std::unordered_set<std::string> canonical_seen;
  std::mutex seen_mutex;
  std::mutex out_mutex;
};

struct RepresentableRecord {
  int rank = 0;
  std::vector<std::array<uint8_t, kMaxN>> cols;
  std::vector<int> bases;
  int witness = 0;
};

struct SparsePavingRecord {
  int rank = 0;
  std::vector<int> circuit_hyperplanes;
  std::vector<int> bases;
  int overlap_bound = 0;
};

uint64_t SplitMix64(uint64_t x) {
  x += kSplitMixIncrement;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

class TrialRng {
 public:
  explicit TrialRng(uint64_t seed) : state_(seed) {}
  uint64_t NextU64() {
    state_ = SplitMix64(state_);
    return state_;
  }
  int UniformInt(int lo, int hi) {
    const uint64_t span = static_cast<uint64_t>(hi - lo + 1);
    return lo + static_cast<int>(NextU64() % span);
  }
  double Uniform01() {
    const double denom = 1.0 / static_cast<double>(std::numeric_limits<uint64_t>::max());
    return static_cast<double>(NextU64()) * denom;
  }

 private:
  uint64_t state_;
};

std::string ModeToString(GenerationMode mode) {
  return mode == GenerationMode::kRepresentable ? "representable" : "sparse_paving";
}

bool ParseModeValue(const std::string& raw, GenerationMode* out) {
  std::string value;
  value.reserve(raw.size());
  for (char ch : raw) {
    value.push_back(ch == '-' ? '_' : static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
  }
  if (value == "representable") {
    *out = GenerationMode::kRepresentable;
    return true;
  }
  if (value == "sparse_paving") {
    *out = GenerationMode::kSparsePaving;
    return true;
  }
  return false;
}

static inline std::string Trim(std::string s) {
  auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [&](char ch) { return !is_space(static_cast<unsigned char>(ch)); }));
  s.erase(std::find_if(s.rbegin(), s.rend(), [&](char ch) { return !is_space(static_cast<unsigned char>(ch)); }).base(), s.end());
  return s;
}

static inline void StripComment(std::string& s) {
  const auto pos = s.find('#');
  if (pos != std::string::npos) {
    s = s.substr(0, pos);
  }
}

static inline std::string StripQuotes(const std::string& s) {
  if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\''))) {
    return s.substr(1, s.size() - 2);
  }
  return s;
}

bool ParseBoolValue(const std::string& raw, bool* out) {
  std::string value;
  value.reserve(raw.size());
  for (char ch : raw) {
    value.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
  }
  if (value == "true" || value == "1") {
    *out = true;
    return true;
  }
  if (value == "false" || value == "0") {
    *out = false;
    return true;
  }
  return false;
}

bool ApplyConfigFile(GenConfig& cfg, const std::string& path, std::string* err) {
  std::ifstream in(path);
  if (!in) {
    if (err) {
      *err = "Could not open config file: " + path;
    }
    return false;
  }
  std::string section;
  std::string line;
  while (std::getline(in, line)) {
    StripComment(line);
    line = Trim(line);
    if (line.empty()) {
      continue;
    }
    if (line.front() == '[' && line.back() == ']') {
      section = Trim(line.substr(1, line.size() - 2));
      continue;
    }
    const auto eq = line.find('=');
    if (eq == std::string::npos || section != "generation") {
      continue;
    }
    std::string key = Trim(line.substr(0, eq));
    std::string value = Trim(line.substr(eq + 1));
    try {
      if (key == "mode") {
        GenerationMode parsed = GenerationMode::kRepresentable;
        if (!ParseModeValue(StripQuotes(value), &parsed)) {
          if (err) *err = "Invalid mode in config: " + value;
          return false;
        }
        cfg.mode = parsed;
      } else if (key == "field") {
        cfg.field = std::stoi(value);
      } else if (key == "rank_min") {
        cfg.rank_min = std::stoi(value);
      } else if (key == "rank_max") {
        cfg.rank_max = std::stoi(value);
      } else if (key == "n") {
        cfg.n = std::stoi(value);
      } else if (key == "threads") {
        cfg.threads = std::stoi(value);
      } else if (key == "seed") {
        cfg.seed = static_cast<uint64_t>(std::stoull(value));
      } else if (key == "max_seconds_total" || key == "max_seconds") {
        cfg.max_seconds = std::stoi(value);
      } else if (key == "max_trials") {
        cfg.max_trials = static_cast<uint64_t>(std::stoull(value));
      } else if (key == "trial_start") {
        cfg.trial_start = static_cast<uint64_t>(std::stoull(value));
      } else if (key == "trial_stride") {
        cfg.trial_stride = static_cast<uint64_t>(std::stoull(value));
      } else if (key == "sparse_accept_prob") {
        cfg.sparse_accept_prob = std::stod(value);
      } else if (key == "sparse_min_circuit_hyperplanes" || key == "sparse_min_ch") {
        cfg.sparse_min_ch = std::stoi(value);
      } else if (key == "sparse_max_circuit_hyperplanes" || key == "sparse_max_ch") {
        cfg.sparse_max_ch = std::stoi(value);
      } else if (key == "require_connected") {
        bool parsed = false;
        if (!ParseBoolValue(value, &parsed)) {
          if (err) *err = "Invalid bool for generation.require_connected: " + value;
          return false;
        }
        cfg.require_connected = parsed;
      } else if (key == "out") {
        cfg.out_path = StripQuotes(value);
      }
    } catch (const std::exception& ex) {
      if (err) *err = "Invalid value in config for key '" + key + "': " + ex.what();
      return false;
    }
  }
  return true;
}

bool ParseArgs(int argc, char** argv, GenConfig& cfg, std::string* err) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const std::string& name) -> std::optional<std::string> {
      if (i + 1 >= argc) {
        if (err) *err = "Missing value for argument: " + name;
        return std::nullopt;
      }
      return std::string(argv[++i]);
    };
    if (arg == "--config") {
      auto val = require_value(arg);
      if (!val) return false;
      cfg.config_path = *val;
      if (!ApplyConfigFile(cfg, cfg.config_path, err)) return false;
    } else if (arg == "--mode") {
      auto val = require_value(arg);
      if (!val) return false;
      GenerationMode parsed = GenerationMode::kRepresentable;
      if (!ParseModeValue(*val, &parsed)) {
        if (err) *err = "Invalid --mode. Use representable or sparse-paving.";
        return false;
      }
      cfg.mode = parsed;
    } else if (arg == "--field") {
      auto val = require_value(arg);
      if (!val) return false;
      cfg.field = std::stoi(*val);
    } else if (arg == "--rank-min") {
      auto val = require_value(arg);
      if (!val) return false;
      cfg.rank_min = std::stoi(*val);
    } else if (arg == "--rank-max") {
      auto val = require_value(arg);
      if (!val) return false;
      cfg.rank_max = std::stoi(*val);
    } else if (arg == "--n") {
      auto val = require_value(arg);
      if (!val) return false;
      cfg.n = std::stoi(*val);
    } else if (arg == "--threads") {
      auto val = require_value(arg);
      if (!val) return false;
      cfg.threads = std::stoi(*val);
    } else if (arg == "--seed") {
      auto val = require_value(arg);
      if (!val) return false;
      cfg.seed = static_cast<uint64_t>(std::stoull(*val));
    } else if (arg == "--max-seconds") {
      auto val = require_value(arg);
      if (!val) return false;
      cfg.max_seconds = std::stoi(*val);
    } else if (arg == "--max-trials") {
      auto val = require_value(arg);
      if (!val) return false;
      cfg.max_trials = static_cast<uint64_t>(std::stoull(*val));
    } else if (arg == "--trial-start") {
      auto val = require_value(arg);
      if (!val) return false;
      cfg.trial_start = static_cast<uint64_t>(std::stoull(*val));
    } else if (arg == "--trial-stride") {
      auto val = require_value(arg);
      if (!val) return false;
      cfg.trial_stride = static_cast<uint64_t>(std::stoull(*val));
    } else if (arg == "--sparse-accept-prob") {
      auto val = require_value(arg);
      if (!val) return false;
      cfg.sparse_accept_prob = std::stod(*val);
    } else if (arg == "--sparse-min-ch") {
      auto val = require_value(arg);
      if (!val) return false;
      cfg.sparse_min_ch = std::stoi(*val);
    } else if (arg == "--sparse-max-ch") {
      auto val = require_value(arg);
      if (!val) return false;
      cfg.sparse_max_ch = std::stoi(*val);
    } else if (arg == "--out") {
      auto val = require_value(arg);
      if (!val) return false;
      cfg.out_path = *val;
    } else if (arg == "--require-connected") {
      cfg.require_connected = true;
    } else if (arg == "--allow-disconnected") {
      cfg.require_connected = false;
    } else if (arg == "--stats-out") {
      auto val = require_value(arg);
      if (!val) return false;
      cfg.stats_out_path = *val;
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: gen_nonpaving [--config path] [--mode representable|sparse-paving] [--field 2|3]\n"
          << "                     [--rank-min 4] [--rank-max 9] [--n 10] [--threads N]\n"
          << "                     [--seed S] [--max-seconds T] [--max-trials K]\n"
          << "                     [--trial-start S] [--trial-stride D]\n"
          << "                     [--sparse-accept-prob P] [--sparse-min-ch K] [--sparse-max-ch K]\n"
          << "                     [--require-connected|--allow-disconnected]\n"
          << "                     [--out path] [--stats-out path]\n";
      std::exit(0);
    } else {
      if (err) *err = "Unknown argument: " + arg;
      return false;
    }
  }
  return true;
}

bool ValidateConfig(const GenConfig& cfg, std::string* err) {
  if (cfg.mode == GenerationMode::kRepresentable && !(cfg.field == 2 || cfg.field == 3)) {
    if (err) *err = "--field must be 2 or 3 in representable mode";
    return false;
  }
  if (cfg.n != 10) {
    if (err) *err = "--n must be 10 for this project";
    return false;
  }
  if (cfg.rank_min < 1 || cfg.rank_max > 9 || cfg.rank_min > cfg.rank_max) {
    if (err) *err = "Invalid rank range";
    return false;
  }
  if (cfg.threads < 1) {
    if (err) *err = "--threads must be >= 1";
    return false;
  }
  if (cfg.max_seconds < 1) {
    if (err) *err = "--max-seconds must be >= 1";
    return false;
  }
  if (cfg.trial_stride < 1) {
    if (err) *err = "--trial-stride must be >= 1";
    return false;
  }
  if (cfg.sparse_accept_prob < 0.0 || cfg.sparse_accept_prob > 1.0) {
    if (err) *err = "--sparse-accept-prob must be in [0,1]";
    return false;
  }
  if (cfg.sparse_min_ch < 0 || cfg.sparse_max_ch < 0) {
    if (err) *err = "--sparse-min-ch and --sparse-max-ch must be >= 0";
    return false;
  }
  if (cfg.sparse_max_ch > 0 && cfg.sparse_max_ch < cfg.sparse_min_ch) {
    if (err) *err = "--sparse-max-ch must be 0 or >= --sparse-min-ch";
    return false;
  }
  return true;
}

std::vector<std::vector<int>> BuildSubsetsBySize(int n) {
  const int total = 1 << n;
  std::vector<std::vector<int>> by_size(n + 1);
  by_size[0].push_back(0);
  for (int mask = 1; mask < total; ++mask) {
    by_size[__builtin_popcount(static_cast<unsigned int>(mask))].push_back(mask);
  }
  return by_size;
}

int EncodeColumn(const std::array<uint8_t, kMaxN>& col, int field, int rank) {
  int mult = 1;
  int encoded = 0;
  for (int r = 0; r < rank; ++r) {
    encoded += mult * static_cast<int>(col[r]);
    mult *= field;
  }
  return encoded;
}

uint16_t EncodeColumnBits(const std::array<uint8_t, kMaxN>& col, int rank) {
  uint16_t out = 0;
  for (int r = 0; r < rank; ++r) {
    if (col[r] & 1U) {
      out |= static_cast<uint16_t>(1U << r);
    }
  }
  return out;
}

int RankGF2Subset(const std::vector<uint16_t>& cols_bits, int rank, int subset_mask) {
  uint16_t basis[kMaxN] = {};
  int out_rank = 0;
  for (int c = 0; c < static_cast<int>(cols_bits.size()); ++c) {
    if (((subset_mask >> c) & 1) == 0) continue;
    uint16_t v = cols_bits[c];
    for (int bit = rank - 1; bit >= 0; --bit) {
      if (((v >> bit) & 1U) == 0) continue;
      if (basis[bit] == 0) {
        basis[bit] = v;
        ++out_rank;
        break;
      }
      v ^= basis[bit];
    }
  }
  return out_rank;
}

int InvModPrime(int x, int q) {
  x %= q;
  if (x < 0) x += q;
  if (x == 0) return 0;
  if (q == 2) return 1;
  if (q == 3) return x == 1 ? 1 : 2;
  return 0;
}

int RankGFqSubset(const std::vector<std::array<uint8_t, kMaxN>>& cols, int field, int rank, int subset_mask) {
  int k = __builtin_popcount(static_cast<unsigned int>(subset_mask));
  if (k == 0) return 0;
  int a[kMaxN][kMaxN] = {};
  int ci = 0;
  for (int c = 0; c < static_cast<int>(cols.size()); ++c) {
    if (((subset_mask >> c) & 1) == 0) continue;
    for (int r = 0; r < rank; ++r) {
      a[r][ci] = static_cast<int>(cols[c][r]);
    }
    ++ci;
  }
  int rr = 0;
  for (int col = 0; col < k && rr < rank; ++col) {
    int pivot = -1;
    for (int r = rr; r < rank; ++r) {
      if (a[r][col] % field != 0) {
        pivot = r;
        break;
      }
    }
    if (pivot < 0) continue;
    if (pivot != rr) {
      for (int j = col; j < k; ++j) {
        std::swap(a[pivot][j], a[rr][j]);
      }
    }
    int inv = InvModPrime(a[rr][col], field);
    for (int j = col; j < k; ++j) {
      a[rr][j] = (a[rr][j] * inv) % field;
    }
    for (int r = 0; r < rank; ++r) {
      if (r == rr) continue;
      int factor = a[r][col] % field;
      if (factor == 0) continue;
      for (int j = col; j < k; ++j) {
        int value = (a[r][j] - factor * a[rr][j]) % field;
        if (value < 0) value += field;
        a[r][j] = value;
      }
    }
    ++rr;
  }
  return rr;
}

int RankSubset(const std::vector<std::array<uint8_t, kMaxN>>& cols, const std::vector<uint16_t>& cols_bits, int field,
               int rank, int subset_mask) {
  return field == 2 ? RankGF2Subset(cols_bits, rank, subset_mask) : RankGFqSubset(cols, field, rank, subset_mask);
}

std::optional<int> FindDependencyWitness(const std::vector<std::array<uint8_t, kMaxN>>& cols,
                                         const std::vector<uint16_t>& cols_bits, int field, int rank,
                                         const std::vector<std::vector<int>>& subsets_by_size) {
  for (int s = 1; s < rank; ++s) {
    for (int mask : subsets_by_size[s]) {
      if (RankSubset(cols, cols_bits, field, rank, mask) < s) {
        return mask;
      }
    }
  }
  return std::nullopt;
}

std::vector<int> EnumerateBasesRepresentable(const std::vector<std::array<uint8_t, kMaxN>>& cols,
                                             const std::vector<uint16_t>& cols_bits, int field, int rank,
                                             const std::vector<std::vector<int>>& subsets_by_size) {
  std::vector<int> bases;
  bases.reserve(subsets_by_size[rank].size());
  for (int mask : subsets_by_size[rank]) {
    if (RankSubset(cols, cols_bits, field, rank, mask) == rank) {
      bases.push_back(mask);
    }
  }
  return bases;
}

int RankFromBasesSubset(const std::vector<int>& bases, int subset_mask) {
  int best = 0;
  for (int base : bases) {
    const int overlap = __builtin_popcount(static_cast<unsigned int>(base & subset_mask));
    if (overlap > best) best = overlap;
  }
  return best;
}

bool IsConnectedRepresentable(const std::vector<std::array<uint8_t, kMaxN>>& cols, const std::vector<uint16_t>& cols_bits,
                              int field, int rank, int n) {
  const int full_mask = (1 << n) - 1;
  for (int mask = 1; mask < full_mask; ++mask) {
    const int comp = full_mask ^ mask;
    if (mask > comp) continue;
    if (RankSubset(cols, cols_bits, field, rank, mask) + RankSubset(cols, cols_bits, field, rank, comp) == rank) {
      return false;
    }
  }
  return true;
}

bool IsConnectedFromBases(const std::vector<int>& bases, int rank, int n) {
  const int full_mask = (1 << n) - 1;
  for (int mask = 1; mask < full_mask; ++mask) {
    const int comp = full_mask ^ mask;
    if (mask > comp) continue;
    if (RankFromBasesSubset(bases, mask) + RankFromBasesSubset(bases, comp) == rank) {
      return false;
    }
  }
  return true;
}

uint64_t Fnv1a64(const std::string& s) {
  uint64_t h = 1469598103934665603ULL;
  for (unsigned char c : s) {
    h ^= static_cast<uint64_t>(c);
    h *= 1099511628211ULL;
  }
  return h;
}

std::string HexU64(uint64_t value) {
  std::ostringstream oss;
  oss << std::hex << std::setfill('0') << std::setw(16) << value;
  return oss.str();
}

std::string CanonicalLabelFromIncidence(int n, const std::vector<int>& bases) {
  const int num_base_nodes = static_cast<int>(bases.size());
  const int nv = n + num_base_nodes;
  const int m = SETWORDSNEEDED(nv);
  std::vector<graph> g(static_cast<size_t>(nv) * static_cast<size_t>(m));
  EMPTYGRAPH(g.data(), m, nv);
  for (int j = 0; j < num_base_nodes; ++j) {
    const int bv = n + j;
    const int mask = bases[j];
    for (int e = 0; e < n; ++e) {
      if ((mask >> e) & 1) {
        ADDONEEDGE(g.data(), e, bv, m);
      }
    }
  }
  std::vector<int> lab(nv), ptn(nv), orbits(nv);
  for (int i = 0; i < nv; ++i) {
    lab[i] = i;
    ptn[i] = 1;
  }
  if (n > 0) ptn[n - 1] = 0;
  if (nv > n) ptn[nv - 1] = 0;
  DEFAULTOPTIONS_GRAPH(options);
  options.getcanon = TRUE;
  options.defaultptn = FALSE;
  statsblk stats{};
  std::vector<setword> workspace(static_cast<size_t>(50) * static_cast<size_t>(m) + 64U);
  std::vector<graph> canon(static_cast<size_t>(nv) * static_cast<size_t>(m));
  nauty(g.data(), lab.data(), ptn.data(), nullptr, orbits.data(), &options, &stats, workspace.data(),
        static_cast<int>(workspace.size()), m, nv, canon.data());
  std::ostringstream oss;
  oss << nv << "|";
  for (int v = 0; v < nv; ++v) {
    const setword* row = GRAPHROW(canon.data(), v, m);
    for (int u = v + 1; u < nv; ++u) {
      if (ISELEMENT(row, u)) {
        oss << v << "-" << u << ",";
      }
    }
  }
  return oss.str();
}

std::string JsonIntArray(const std::vector<int>& values) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) oss << ",";
    oss << values[i];
  }
  oss << "]";
  return oss.str();
}

uint64_t TrialSeed(const GenConfig& cfg, uint64_t trial_id) {
  const uint64_t mode_tag =
      cfg.mode == GenerationMode::kRepresentable ? 0xa24baed4963ee407ULL : 0xf6ef87f5cc98fc11ULL;
  return SplitMix64(cfg.seed ^ mode_tag ^ (static_cast<uint64_t>(cfg.field) * kFieldStride) ^
                    (trial_id * kSplitMixIncrement));
}

void ShuffleInPlace(std::vector<int>& values, TrialRng& rng) {
  for (int i = static_cast<int>(values.size()) - 1; i > 0; --i) {
    std::swap(values[static_cast<size_t>(i)], values[static_cast<size_t>(rng.UniformInt(0, i))]);
  }
}

bool BuildRepresentableCandidate(const GenConfig& cfg, uint64_t trial_id, const std::vector<std::vector<int>>& subsets_by_size,
                                 RepresentableRecord* out_record, Stats& stats) {
  TrialRng rng(TrialSeed(cfg, trial_id));
  const int rank = rng.UniformInt(cfg.rank_min, cfg.rank_max);
  const int full_mask = (1 << cfg.n) - 1;
  std::vector<std::array<uint8_t, kMaxN>> cols(cfg.n);
  for (int c = 0; c < cfg.n; ++c) {
    for (int r = 0; r < rank; ++r) {
      cols[c][r] = static_cast<uint8_t>(rng.UniformInt(0, cfg.field - 1));
    }
  }
  std::vector<uint16_t> cols_bits;
  if (cfg.field == 2) {
    cols_bits.resize(cfg.n);
    for (int c = 0; c < cfg.n; ++c) {
      cols_bits[c] = EncodeColumnBits(cols[c], rank);
    }
  }
  if (RankSubset(cols, cols_bits, cfg.field, rank, full_mask) < rank) {
    return false;
  }
  stats.full_rank.fetch_add(1);
  auto witness = FindDependencyWitness(cols, cols_bits, cfg.field, rank, subsets_by_size);
  if (!witness) {
    return false;
  }
  stats.non_paving.fetch_add(1);
  auto bases = EnumerateBasesRepresentable(cols, cols_bits, cfg.field, rank, subsets_by_size);
  if (cfg.require_connected && !IsConnectedRepresentable(cols, cols_bits, cfg.field, rank, cfg.n)) {
    stats.disconnected_filtered.fetch_add(1);
    return false;
  }
  out_record->rank = rank;
  out_record->cols = std::move(cols);
  out_record->bases = std::move(bases);
  out_record->witness = *witness;
  return true;
}

bool BuildSparsePavingCandidate(const GenConfig& cfg, uint64_t trial_id, const std::vector<std::vector<int>>& subsets_by_size,
                                SparsePavingRecord* out_record, Stats& stats) {
  TrialRng rng(TrialSeed(cfg, trial_id));
  const int rank = rng.UniformInt(cfg.rank_min, cfg.rank_max);
  const int overlap_bound = std::max(0, rank - 2);
  std::vector<int> rank_subsets = subsets_by_size[rank];
  ShuffleInPlace(rank_subsets, rng);
  std::vector<int> circuit_hyperplanes;
  for (int mask : rank_subsets) {
    if (cfg.sparse_max_ch > 0 && static_cast<int>(circuit_hyperplanes.size()) >= cfg.sparse_max_ch) {
      break;
    }
    if (rng.Uniform01() > cfg.sparse_accept_prob) {
      continue;
    }
    bool allowed = true;
    for (int chosen : circuit_hyperplanes) {
      if (__builtin_popcount(static_cast<unsigned int>(mask & chosen)) > overlap_bound) {
        allowed = false;
        break;
      }
    }
    if (allowed) {
      circuit_hyperplanes.push_back(mask);
    }
  }
  if (static_cast<int>(circuit_hyperplanes.size()) < cfg.sparse_min_ch) {
    stats.sparse_rejected_min_ch.fetch_add(1);
    return false;
  }
  std::unordered_set<int> ch_set;
  ch_set.reserve(circuit_hyperplanes.size() * 2 + 1);
  for (int mask : circuit_hyperplanes) {
    ch_set.insert(mask);
  }
  std::vector<int> bases;
  bases.reserve(subsets_by_size[rank].size());
  for (int mask : subsets_by_size[rank]) {
    if (ch_set.find(mask) == ch_set.end()) {
      bases.push_back(mask);
    }
  }
  if (bases.empty()) {
    stats.sparse_rejected_empty_bases.fetch_add(1);
    return false;
  }
  stats.non_paving.fetch_add(1);
  stats.sparse_circuit_hyperplanes_total.fetch_add(static_cast<uint64_t>(circuit_hyperplanes.size()));
  if (cfg.require_connected && !IsConnectedFromBases(bases, rank, cfg.n)) {
    stats.disconnected_filtered.fetch_add(1);
    return false;
  }
  out_record->rank = rank;
  out_record->circuit_hyperplanes = std::move(circuit_hyperplanes);
  out_record->bases = std::move(bases);
  out_record->overlap_bound = overlap_bound;
  return true;
}

void WriteStats(const GenConfig& cfg, const Stats& stats, double elapsed_seconds) {
  if (cfg.stats_out_path.empty()) {
    return;
  }
  std::filesystem::path p(cfg.stats_out_path);
  if (!p.parent_path().empty()) {
    std::filesystem::create_directories(p.parent_path());
  }
  std::ofstream out(cfg.stats_out_path, std::ios::trunc);
  if (!out) {
    std::cerr << "Could not write stats file: " << cfg.stats_out_path << "\n";
    return;
  }
  out << "{\n";
  out << "  \"mode\": \"" << ModeToString(cfg.mode) << "\",\n";
  if (cfg.mode == GenerationMode::kRepresentable) {
    out << "  \"field\": " << cfg.field << ",\n";
  }
  out << "  \"rank_min\": " << cfg.rank_min << ",\n";
  out << "  \"rank_max\": " << cfg.rank_max << ",\n";
  out << "  \"n\": " << cfg.n << ",\n";
  out << "  \"seed\": " << cfg.seed << ",\n";
  out << "  \"trial_start\": " << cfg.trial_start << ",\n";
  out << "  \"trial_stride\": " << cfg.trial_stride << ",\n";
  out << "  \"elapsed_seconds\": " << elapsed_seconds << ",\n";
  out << "  \"candidates\": " << stats.candidates.load() << ",\n";
  out << "  \"full_rank\": " << stats.full_rank.load() << ",\n";
  out << "  \"non_paving\": " << stats.non_paving.load() << ",\n";
  out << "  \"disconnected_filtered\": " << stats.disconnected_filtered.load() << ",\n";
  out << "  \"unique_hits\": " << stats.unique_hits.load() << ",\n";
  out << "  \"duplicates\": " << stats.duplicates.load() << ",\n";
  out << "  \"emitted_bases\": " << stats.emitted_bases.load() << ",\n";
  out << "  \"sparse_rejected_min_ch\": " << stats.sparse_rejected_min_ch.load() << ",\n";
  out << "  \"sparse_rejected_empty_bases\": " << stats.sparse_rejected_empty_bases.load() << ",\n";
  out << "  \"sparse_circuit_hyperplanes_total\": " << stats.sparse_circuit_hyperplanes_total.load() << "\n";
  out << "}\n";
}

void WorkerLoop(const GenConfig& cfg, const std::vector<std::vector<int>>& subsets_by_size, Stats& stats, SharedState& shared,
                std::ofstream& out_stream, std::atomic<uint64_t>& trial_counter,
                const std::chrono::steady_clock::time_point start_time) {
  while (true) {
    const uint64_t local_idx = trial_counter.fetch_add(1);
    if (cfg.max_trials > 0 && local_idx >= cfg.max_trials) {
      return;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time).count();
    if (elapsed >= cfg.max_seconds) {
      return;
    }
    const uint64_t trial_id = cfg.trial_start + local_idx * cfg.trial_stride;
    stats.candidates.fetch_add(1);

    std::vector<int> bases;
    std::ostringstream line;
    if (cfg.mode == GenerationMode::kRepresentable) {
      RepresentableRecord rec;
      if (!BuildRepresentableCandidate(cfg, trial_id, subsets_by_size, &rec, stats)) {
        continue;
      }
      bases = rec.bases;
      const std::string canonical_label = CanonicalLabelFromIncidence(cfg.n, bases);
      bool is_new = false;
      {
        std::lock_guard<std::mutex> lock(shared.seen_mutex);
        is_new = shared.canonical_seen.insert(canonical_label).second;
      }
      if (!is_new) {
        stats.duplicates.fetch_add(1);
        continue;
      }
      std::vector<int> matrix_cols;
      matrix_cols.reserve(cfg.n);
      for (int c = 0; c < cfg.n; ++c) {
        matrix_cols.push_back(EncodeColumn(rec.cols[c], cfg.field, rec.rank));
      }
      line << "{\"id\":\"" << HexU64(Fnv1a64(canonical_label)) << "\"";
      line << ",\"generator_mode\":\"representable\"";
      line << ",\"field\":" << cfg.field;
      line << ",\"rank\":" << rec.rank;
      line << ",\"n\":" << cfg.n;
      line << ",\"seed\":" << cfg.seed;
      line << ",\"trial\":" << trial_id;
      line << ",\"matrix_cols\":" << JsonIntArray(matrix_cols);
      line << ",\"bases\":" << JsonIntArray(rec.bases);
      line << ",\"non_paving_witness\":" << rec.witness;
      line << "}\n";
      stats.emitted_bases.fetch_add(static_cast<uint64_t>(rec.bases.size()));
    } else {
      SparsePavingRecord rec;
      if (!BuildSparsePavingCandidate(cfg, trial_id, subsets_by_size, &rec, stats)) {
        continue;
      }
      bases = rec.bases;
      const std::string canonical_label = CanonicalLabelFromIncidence(cfg.n, bases);
      bool is_new = false;
      {
        std::lock_guard<std::mutex> lock(shared.seen_mutex);
        is_new = shared.canonical_seen.insert(canonical_label).second;
      }
      if (!is_new) {
        stats.duplicates.fetch_add(1);
        continue;
      }
      line << "{\"id\":\"" << HexU64(Fnv1a64(canonical_label)) << "\"";
      line << ",\"generator_mode\":\"sparse_paving\"";
      line << ",\"rank\":" << rec.rank;
      line << ",\"n\":" << cfg.n;
      line << ",\"seed\":" << cfg.seed;
      line << ",\"trial\":" << trial_id;
      line << ",\"bases\":" << JsonIntArray(rec.bases);
      line << ",\"circuit_hyperplanes\":" << JsonIntArray(rec.circuit_hyperplanes);
      line << ",\"sparse_overlap_bound\":" << rec.overlap_bound;
      line << "}\n";
      stats.emitted_bases.fetch_add(static_cast<uint64_t>(rec.bases.size()));
    }

    {
      std::lock_guard<std::mutex> lock(shared.out_mutex);
      out_stream << line.str();
      out_stream.flush();
    }
    stats.unique_hits.fetch_add(1);
  }
}

}  // namespace

int main(int argc, char** argv) {
  GenConfig cfg;
  if (cfg.threads < 1) {
    cfg.threads = std::max(1u, std::thread::hardware_concurrency());
  }
  std::string err;
  if (!ParseArgs(argc, argv, cfg, &err)) {
    std::cerr << err << "\n";
    return kExitUsageError;
  }
  if (!ValidateConfig(cfg, &err)) {
    std::cerr << err << "\n";
    return kExitUsageError;
  }
  std::filesystem::path out_path(cfg.out_path);
  if (!out_path.parent_path().empty()) {
    std::filesystem::create_directories(out_path.parent_path());
  }
  std::ofstream out(cfg.out_path, std::ios::trunc);
  if (!out) {
    std::cerr << "Could not open output: " << cfg.out_path << "\n";
    return kExitRuntimeError;
  }

  Stats stats;
  SharedState shared;
  std::atomic<uint64_t> trial_counter{0};
  auto subsets_by_size = BuildSubsetsBySize(cfg.n);
  const auto start = std::chrono::steady_clock::now();
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(cfg.threads));
  for (int worker = 0; worker < cfg.threads; ++worker) {
    workers.emplace_back(WorkerLoop, std::cref(cfg), std::cref(subsets_by_size), std::ref(stats), std::ref(shared),
                         std::ref(out), std::ref(trial_counter), start);
  }
  for (auto& t : workers) {
    t.join();
  }
  const auto end = std::chrono::steady_clock::now();
  const double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
  WriteStats(cfg, stats, elapsed_seconds);

  std::cerr << "Generation complete."
            << " mode=" << ModeToString(cfg.mode)
            << " candidates=" << stats.candidates.load()
            << " full_rank=" << stats.full_rank.load()
            << " non_paving=" << stats.non_paving.load()
            << " disconnected_filtered=" << stats.disconnected_filtered.load()
            << " unique_hits=" << stats.unique_hits.load()
            << " elapsed_seconds=" << elapsed_seconds << "\n";
  return 0;
}
