#pragma once
#include <vector>

namespace gpumap {

// Simple Bloom Filter implementation for avoiding duplicate neighbor checks
template <typename Idx_t = size_t> class BloomFilter {
public:
  BloomFilter(size_t nrow, size_t num_sets_per_list, size_t num_hashes)
      : nrow_(nrow), num_sets_per_list_(num_sets_per_list),
        num_hashes_(num_hashes),
        bitsets_(nrow_ * num_sets_per_list_ * num_bits_per_set_) {};

  void set_nrow(size_t nrow) { nrow_ = nrow; }

  void clear() {
    if (is_cleared_)
      return;
#pragma omp parallel for
    for (size_t i = 0; i < nrow_ * num_sets_per_list_ * num_bits_per_set_;
         ++i) {
      bitsets_[i] = 0;
    }
    is_cleared_ = true;
  }

  bool check(size_t list_id, Idx_t key) const {
    bool is_present = true;
    uint32_t hash = hash0(static_cast<uint32_t>(key));
    size_t global_set_idx = list_id * num_sets_per_list_ +
                            (static_cast<size_t>(key) % num_sets_per_list_);

    is_present &= bitsets_[global_set_idx + hash % num_bits_per_set_];

    if (!is_present)
      return false;

    for (size_t i = 1; i < num_hashes_; ++i) {
      hash = hash + hash1(static_cast<uint32_t>(key));
      is_present &= bitsets_[global_set_idx + hash % num_bits_per_set_];
      if (!is_present)
        return false;
    }
    return true;
  }

  void add(size_t list_id, Idx_t key) {
    if (is_cleared_)
      is_cleared_ = false;

    const uint32_t k = static_cast<uint32_t>(key);
    uint32_t hash = hash0(k);
    const size_t global_set_idx =
        list_id * num_bits_per_set_ * num_sets_per_list_ +
        (k % num_sets_per_list_) * num_bits_per_set_;

    bitsets_[global_set_idx + (hash % num_bits_per_set_)] = 1;
    for (size_t i = 1; i < num_hashes_; ++i) {
      hash = hash + hash1(k);
      bitsets_[global_set_idx + (hash % num_bits_per_set_)] = 1;
    }
  }

private:
  uint32_t hash0(uint32_t value) {
    value *= 1103515245;
    value += 12345;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    return value;
  }

  uint32_t hash1(uint32_t value) {
    value *= 1664525;
    value += 1013904223;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    return value;
  }

  static constexpr int num_bits_per_set_ = 512;
  bool is_cleared_ = true;

  std::vector<bool> bitsets_;
  size_t nrow_;
  size_t num_sets_per_list_;
  size_t num_hashes_;
};
} // namespace gpumap