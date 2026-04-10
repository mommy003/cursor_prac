#include "quantizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <zlib.h>

// .eigen.q8e.bin: same header + scales as .eigen.q8.bin, then uint64 uncompressed_len,
// uint64 compressed_len, zlib DEFLATE payload (compress()) of the column-major int8 matrix.

namespace eigen_quantize {

const char* kInputSuffix = ".eigen.bin";

const char* output_suffix(const QuantizationOptions& options) {
    if (options.q_per_snp_column) {
        if (options.bits == 4) return ".eigen.q4qc.bin";
        if (options.bits == 8 && options.entropy_coding) return ".eigen.q8eqc.bin";
        if (options.bits == 8) return ".eigen.q8qc.bin";
        if (options.bits == 16) return ".eigen.q16qc.bin";
    }
    if (options.bits == 4) return ".eigen.q4.bin";
    if (options.bits == 8 && options.entropy_coding) return ".eigen.q8e.bin";
    if (options.bits == 8) return ".eigen.q8.bin";
    return ".eigen.q16.bin";
}

float quantization_bound(int bits) {
    if (bits == 4) return 7.0f;
    if (bits == 8) return 127.0f;
    return 32767.0f;
}

void copy_info_file(const fs::path& input_dir, const fs::path& output_dir, const string& filename) {
    const fs::path input_path = input_dir / filename;
    if (!fs::exists(input_path)) return;
    fs::copy_file(input_path, output_dir / filename, fs::copy_options::overwrite_existing);
}

int16_t quantize_value(float value, float scale, int bits) {
    if (scale <= 0.0f) {
        return 0;
    }

    const float bound = quantization_bound(bits);
    const float scaled = value / scale * bound;
    const long rounded = lround(scaled);
    const long limit = static_cast<long>(bound);
    const long clamped = max<long>(-limit, min<long>(limit, rounded));
    return static_cast<int16_t>(clamped);
}

static void write_zlib_matrix(FILE* out, const vector<int8_t>& raw, const fs::path& output_path) {
    const uLong src_len = static_cast<uLong>(raw.size());
    uLong dst_cap = compressBound(src_len);
    vector<Bytef> compressed(dst_cap);
    uLong dst_len = dst_cap;
    const int zrc = compress(compressed.data(), &dst_len, reinterpret_cast<const Bytef*>(raw.data()), src_len);
    if (zrc != Z_OK) {
        throw runtime_error("zlib compress failed for " + output_path.string());
    }
    const uint64_t uncompressed_size = static_cast<uint64_t>(raw.size());
    const uint64_t compressed_size = static_cast<uint64_t>(dst_len);
    if (fwrite(&uncompressed_size, sizeof(uint64_t), 1, out) != 1) {
        throw runtime_error("Write error (uncompressed size) in " + output_path.string());
    }
    if (fwrite(&compressed_size, sizeof(uint64_t), 1, out) != 1) {
        throw runtime_error("Write error (compressed size) in " + output_path.string());
    }
    if (fwrite(compressed.data(), 1, dst_len, out) != dst_len) {
        throw runtime_error("Write error (compressed matrix) in " + output_path.string());
    }
}

/** Q = diag(sqrt(lambda)) * U' (k x m, column-major); one scale per SNP column. */
static void quantize_file_q_per_snp_column(const fs::path& input_dir,
                                         const fs::path& output_dir,
                                         const string& name,
                                         const QuantizationOptions& options,
                                         QuantizationSummary& summary) {
    const fs::path input_path = input_dir / (name + kInputSuffix);
    const fs::path output_path = output_dir / (name + output_suffix(options));

    FILE* fp = fopen(input_path.c_str(), "rb");
    if (!fp) {
        throw runtime_error("Cannot open input file " + input_path.string());
    }
    FILE* out = fopen(output_path.c_str(), "wb");
    if (!out) {
        fclose(fp);
        throw runtime_error("Cannot open output file " + output_path.string());
    }

    try {
        int32_t m = 0;
        int32_t k = 0;
        float sum_pos_eigval = 0.0f;
        float eigen_cutoff = 0.0f;
        if (fread(&m, sizeof(int32_t), 1, fp) != 1) {
            throw runtime_error("Read error (m) in " + input_path.string());
        }
        if (fread(&k, sizeof(int32_t), 1, fp) != 1) {
            throw runtime_error("Read error (k) in " + input_path.string());
        }
        if (fread(&sum_pos_eigval, sizeof(float), 1, fp) != 1) {
            throw runtime_error("Read error (sumPosEigVal) in " + input_path.string());
        }
        if (fread(&eigen_cutoff, sizeof(float), 1, fp) != 1) {
            throw runtime_error("Read error (eigenCutoff) in " + input_path.string());
        }
        if (m < 0 || k < 0) {
            throw runtime_error("Negative matrix dimensions in " + input_path.string());
        }
        vector<float> eigenvalues(k);
        if (fread(eigenvalues.data(), sizeof(float), k, fp) != static_cast<size_t>(k)) {
            throw runtime_error("Read error (eigenvalues) in " + input_path.string());
        }

        vector<float> u(static_cast<size_t>(m) * static_cast<size_t>(k));
        for (int32_t j = 0; j < k; ++j) {
            if (fread(u.data() + static_cast<size_t>(j) * static_cast<size_t>(m), sizeof(float),
                      static_cast<size_t>(m), fp) != static_cast<size_t>(m)) {
                throw runtime_error("Read error (eigenvector column) in " + input_path.string());
            }
        }

        const uint64_t km = static_cast<uint64_t>(k) * static_cast<uint64_t>(m);
        vector<float> Q(km);
        for (int32_t i = 0; i < m; ++i) {
            for (int32_t j = 0; j < k; ++j) {
                const float sq = eigenvalues[j] > 0.0f ? sqrtf(eigenvalues[j]) : 0.0f;
                Q[static_cast<size_t>(j) + static_cast<size_t>(i) * static_cast<size_t>(k)] =
                    sq * u[static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(m)];
            }
        }

        vector<float> scale_snp(m, 0.0f);
        for (int32_t i = 0; i < m; ++i) {
            float s = 0.0f;
            for (int32_t j = 0; j < k; ++j) {
                s = max(s, fabsf(Q[static_cast<size_t>(j) + static_cast<size_t>(i) * static_cast<size_t>(k)]));
            }
            scale_snp[i] = s;
        }

        if (fwrite(&m, sizeof(int32_t), 1, out) != 1) {
            throw runtime_error("Write error (m) in " + output_path.string());
        }
        if (fwrite(&k, sizeof(int32_t), 1, out) != 1) {
            throw runtime_error("Write error (k) in " + output_path.string());
        }
        if (fwrite(&sum_pos_eigval, sizeof(float), 1, out) != 1) {
            throw runtime_error("Write error (sumPosEigVal) in " + output_path.string());
        }
        if (fwrite(&eigen_cutoff, sizeof(float), 1, out) != 1) {
            throw runtime_error("Write error (eigenCutoff) in " + output_path.string());
        }
        if (fwrite(eigenvalues.data(), sizeof(float), k, out) != static_cast<size_t>(k)) {
            throw runtime_error("Write error (eigenvalues) in " + output_path.string());
        }
        if (fwrite(scale_snp.data(), sizeof(float), m, out) != static_cast<size_t>(m)) {
            throw runtime_error("Write error (SNP column scales) in " + output_path.string());
        }

        const bool q8_entropy = (options.bits == 8 && options.entropy_coding);
        vector<int8_t> matrix_q8;
        if (q8_entropy) {
            matrix_q8.resize(km);
        }

        for (int32_t i = 0; i < m; ++i) {
            const float sc = scale_snp[i];
            if (options.bits == 4) {
                const int32_t packed_k = (k + 1) / 2;
                vector<uint8_t> packed(packed_k);
                for (int32_t j = 0; j < k; j += 2) {
                    const float v0 = Q[static_cast<size_t>(j) + static_cast<size_t>(i) * static_cast<size_t>(k)];
                    const float v1 =
                        j + 1 < k
                            ? Q[static_cast<size_t>(j + 1) +
                                static_cast<size_t>(i) * static_cast<size_t>(k)]
                            : 0.0f;
                    const int16_t q0 = quantize_value(v0, sc, options.bits);
                    const int16_t q1 = quantize_value(v1, sc, options.bits);
                    packed[static_cast<size_t>(j / 2)] =
                        static_cast<uint8_t>((q0 & 0x0F) | ((q1 & 0x0F) << 4));
                }
                if (fwrite(packed.data(), sizeof(uint8_t), packed_k, out) != static_cast<size_t>(packed_k)) {
                    throw runtime_error("Write error (quantized Q column) in " + output_path.string());
                }
            } else if (options.bits == 8) {
                if (q8_entropy) {
                    for (int32_t j = 0; j < k; ++j) {
                        const float v = Q[static_cast<size_t>(j) + static_cast<size_t>(i) * static_cast<size_t>(k)];
                        matrix_q8[static_cast<size_t>(j) + static_cast<size_t>(i) * static_cast<size_t>(k)] =
                            static_cast<int8_t>(quantize_value(v, sc, options.bits));
                    }
                } else {
                    vector<int8_t> col_q(k);
                    for (int32_t j = 0; j < k; ++j) {
                        const float v = Q[static_cast<size_t>(j) + static_cast<size_t>(i) * static_cast<size_t>(k)];
                        col_q[static_cast<size_t>(j)] =
                            static_cast<int8_t>(quantize_value(v, sc, options.bits));
                    }
                    if (fwrite(col_q.data(), sizeof(int8_t), k, out) != static_cast<size_t>(k)) {
                        throw runtime_error("Write error (quantized Q column) in " + output_path.string());
                    }
                }
            } else {
                vector<int16_t> col_q(k);
                for (int32_t j = 0; j < k; ++j) {
                    const float v = Q[static_cast<size_t>(j) + static_cast<size_t>(i) * static_cast<size_t>(k)];
                    col_q[static_cast<size_t>(j)] = quantize_value(v, sc, options.bits);
                }
                if (fwrite(col_q.data(), sizeof(int16_t), k, out) != static_cast<size_t>(k)) {
                    throw runtime_error("Write error (quantized Q column) in " + output_path.string());
                }
            }
        }

        if (q8_entropy) {
            write_zlib_matrix(out, matrix_q8, output_path);
        }

        fclose(fp);
        fclose(out);

        const uint64_t original_bytes = fs::file_size(input_path);
        const uint64_t quantized_bytes = fs::file_size(output_path);

        cout << "Quantized (Q per SNP col) " << name << ".eigen.bin"
             << " -> " << name << output_suffix(options) << " (m=" << m << ", k=" << k
             << ", q=" << options.bits << (options.entropy_coding ? ", zlib" : "") << ", bytes "
             << original_bytes << " -> " << quantized_bytes << ")\n"
             << flush;

        summary.total_original_bytes += original_bytes;
        summary.total_quantized_bytes += quantized_bytes;
        summary.num_files += 1;
    } catch (...) {
        fclose(fp);
        fclose(out);
        throw;
    }
}

void quantize_file(const fs::path& input_dir,
                   const fs::path& output_dir,
                   const string& name,
                   const QuantizationOptions& options,
                   QuantizationSummary& summary) {
    if (options.q_per_snp_column) {
        quantize_file_q_per_snp_column(input_dir, output_dir, name, options, summary);
        return;
    }

    const fs::path input_path = input_dir / (name + kInputSuffix);
    const fs::path output_path = output_dir / (name + output_suffix(options));

    FILE* fp = fopen(input_path.c_str(), "rb");
    if (!fp) {
        throw runtime_error("Cannot open input file " + input_path.string());
    }

    FILE* out = fopen(output_path.c_str(), "wb");
    if (!out) {
        fclose(fp);
        throw runtime_error("Cannot open output file " + output_path.string());
    }

    try {
        int32_t num_snps = 0;
        int32_t num_eigenvalues = 0;
        float sum_pos_eigval = 0.0f;
        float eigen_cutoff = 0.0f;

        if (fread(&num_snps, sizeof(int32_t), 1, fp) != 1) {
            throw runtime_error("Read error (m) in " + input_path.string());
        }
        if (fread(&num_eigenvalues, sizeof(int32_t), 1, fp) != 1) {
            throw runtime_error("Read error (k) in " + input_path.string());
        }
        if (fread(&sum_pos_eigval, sizeof(float), 1, fp) != 1) {
            throw runtime_error("Read error (sumPosEigVal) in " + input_path.string());
        }
        if (fread(&eigen_cutoff, sizeof(float), 1, fp) != 1) {
            throw runtime_error("Read error (eigenCutoff) in " + input_path.string());
        }
        if (num_snps < 0 || num_eigenvalues < 0) {
            throw runtime_error("Negative matrix dimensions in " + input_path.string());
        }

        vector<float> eigenvalues(num_eigenvalues);
        if (fread(eigenvalues.data(), sizeof(float), num_eigenvalues, fp) !=
            static_cast<size_t>(num_eigenvalues)) {
            throw runtime_error("Read error (eigenvalues) in " + input_path.string());
        }

        const long matrix_offset = ftell(fp);
        if (matrix_offset < 0) {
            throw runtime_error("Failed to record matrix offset in " + input_path.string());
        }

        vector<float> column(num_snps);
        vector<float> scales(num_eigenvalues, 0.0f);

        for (int32_t col = 0; col < num_eigenvalues; ++col) {
            if (fread(column.data(), sizeof(float), num_snps, fp) !=
                static_cast<size_t>(num_snps)) {
                throw runtime_error("Read error (eigenvector column) in " + input_path.string());
            }

            float scale = 0.0f;
            for (int32_t row = 0; row < num_snps; ++row) {
                scale = max(scale, fabs(column[row]));
            }
            scales[col] = scale;
        }

        if (fseek(fp, matrix_offset, SEEK_SET) != 0) {
            throw runtime_error("Failed to rewind eigenvectors in " + input_path.string());
        }

        if (fwrite(&num_snps, sizeof(int32_t), 1, out) != 1) {
            throw runtime_error("Write error (m) in " + output_path.string());
        }
        if (fwrite(&num_eigenvalues, sizeof(int32_t), 1, out) != 1) {
            throw runtime_error("Write error (k) in " + output_path.string());
        }
        if (fwrite(&sum_pos_eigval, sizeof(float), 1, out) != 1) {
            throw runtime_error("Write error (sumPosEigVal) in " + output_path.string());
        }
        if (fwrite(&eigen_cutoff, sizeof(float), 1, out) != 1) {
            throw runtime_error("Write error (eigenCutoff) in " + output_path.string());
        }
        if (fwrite(eigenvalues.data(), sizeof(float), num_eigenvalues, out) !=
            static_cast<size_t>(num_eigenvalues)) {
            throw runtime_error("Write error (eigenvalues) in " + output_path.string());
        }
        if (fwrite(scales.data(), sizeof(float), num_eigenvalues, out) !=
            static_cast<size_t>(num_eigenvalues)) {
            throw runtime_error("Write error (column scales) in " + output_path.string());
        }

        const bool q8_entropy = (options.bits == 8 && options.entropy_coding);
        vector<int8_t> matrix_q8;
        if (q8_entropy) {
            const uint64_t total = static_cast<uint64_t>(num_snps) * static_cast<uint64_t>(num_eigenvalues);
            matrix_q8.resize(total);
        }

        for (int32_t col = 0; col < num_eigenvalues; ++col) {
            if (fread(column.data(), sizeof(float), num_snps, fp) !=
                static_cast<size_t>(num_snps)) {
                throw runtime_error("Read error (eigenvector column) in " + input_path.string());
            }

            const float scale = scales[col];
            if (options.bits == 4) {
                const int32_t packed_bytes = (num_snps + 1) / 2;
                vector<uint8_t> packed(packed_bytes);
                for (int32_t row = 0; row < num_snps; row += 2) {
                    const int16_t q0 = quantize_value(column[row], scale, options.bits);
                    const int16_t q1 =
                        row + 1 < num_snps
                            ? quantize_value(column[row + 1], scale, options.bits)
                            : 0;
                    packed[row / 2] = static_cast<uint8_t>((q0 & 0x0F) | ((q1 & 0x0F) << 4));
                }
                if (fwrite(packed.data(), sizeof(uint8_t), packed_bytes, out) !=
                    static_cast<size_t>(packed_bytes)) {
                    throw runtime_error("Write error (quantized eigenvector column) in " +
                                        output_path.string());
                }
            } else if (options.bits == 8) {
                if (q8_entropy) {
                    int8_t* col_dst = matrix_q8.data() + static_cast<size_t>(col) * static_cast<size_t>(num_snps);
                    for (int32_t row = 0; row < num_snps; ++row) {
                        col_dst[row] = static_cast<int8_t>(quantize_value(column[row], scale, options.bits));
                    }
                } else {
                    vector<int8_t> quantized_column(num_snps);
                    for (int32_t row = 0; row < num_snps; ++row) {
                        quantized_column[row] =
                            static_cast<int8_t>(quantize_value(column[row], scale, options.bits));
                    }
                    if (fwrite(quantized_column.data(), sizeof(int8_t), num_snps, out) !=
                        static_cast<size_t>(num_snps)) {
                        throw runtime_error("Write error (quantized eigenvector column) in " +
                                            output_path.string());
                    }
                }
            } else {
                vector<int16_t> quantized_column(num_snps);
                for (int32_t row = 0; row < num_snps; ++row) {
                    quantized_column[row] = quantize_value(column[row], scale, options.bits);
                }
                if (fwrite(quantized_column.data(), sizeof(int16_t), num_snps, out) !=
                    static_cast<size_t>(num_snps)) {
                    throw runtime_error("Write error (quantized eigenvector column) in " +
                                        output_path.string());
                }
            }
        }

        if (q8_entropy) {
            write_zlib_matrix(out, matrix_q8, output_path);
        }

        fclose(fp);
        fclose(out);

        const uint64_t original_bytes = fs::file_size(input_path);
        const uint64_t quantized_bytes = fs::file_size(output_path);

        cout << "Quantized " << name << ".eigen.bin"
             << " -> " << name << output_suffix(options)
             << " (m=" << num_snps
             << ", k=" << num_eigenvalues
             << ", q=" << options.bits
             << (options.entropy_coding ? ", zlib" : "")
             << ", bytes " << original_bytes
             << " -> " << quantized_bytes << ")\n" << flush;

        summary.total_original_bytes += original_bytes;
        summary.total_quantized_bytes += quantized_bytes;
        summary.num_files += 1;
        return;
    } catch (...) {
        fclose(fp);
        fclose(out);
        throw;
    }
}

vector<string> list_input_names(const fs::path& input_dir) {
    vector<string> names;
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (!entry.is_regular_file()) continue;

        const string filename = entry.path().filename().string();
        const size_t suffix_len = char_traits<char>::length(kInputSuffix);
        if (filename.size() < suffix_len) continue;
        if (filename.substr(filename.size() - suffix_len) != kInputSuffix) continue;

        names.push_back(filename.substr(0, filename.size() - suffix_len));
    }
    sort(names.begin(), names.end());
    return names;
}

QuantizationSummary quantize_directory(const fs::path& input_dir,
                                       const fs::path& output_dir,
                                       const QuantizationOptions& options) {
    if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
        throw runtime_error("Input folder does not exist: " + input_dir.string());
    }
    if (options.entropy_coding && options.bits != 8) {
        throw runtime_error("--entropy is only supported with 8-bit quantization.");
    }
    if (options.bits != 4 && options.bits != 8 && options.bits != 16) {
        throw runtime_error("Only q4, q8, and q16 are supported.");
    }

    fs::create_directories(output_dir);
    copy_info_file(input_dir, output_dir, "ldm.info");
    copy_info_file(input_dir, output_dir, "snp.info");

    const vector<string> names = list_input_names(input_dir);
    if (names.empty()) {
        throw runtime_error("No .eigen.bin files found in " + input_dir.string());
    }

    QuantizationSummary summary;

    for (const auto& name : names) {
        quantize_file(input_dir, output_dir, name, options, summary);
    }

    return summary;
}

}  // namespace eigen_quantize
