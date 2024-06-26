#include <vector>
#include <iostream>
#include <algorithm>
#include <tuple>
#include <type_traits>

// Fill the zipped vector with pairs consisting of the
// corresponding elements of a and b. (This assumes 
// that the vectors have equal length)
template <typename tuple_type> void zip(
    const bool* a, 
    const float* b,
    const float* sample_weight,
    const size_t len, 
    std::vector<tuple_type> &zipped)
    {
        for(size_t i=0; i<len; ++i)
        {
            if constexpr(std::is_same<tuple_type, std::tuple<bool, float, float>>::value) {
                zipped.push_back(std::make_tuple(a[i], b[i], sample_weight[i]));
            }
            else {
                zipped.push_back(std::make_tuple(a[i], b[i]));
            }
        }
    }

double trapezoid_area(double x1, double x2, double y1, double y2) {
  double dx = x2 - x1;
  double dy = y2 - y1;
  return dx * y1 + dy * dx / 2.0;
}

template <typename tuple_type> float auc_kernel(float* ts, bool* st, size_t len, float* sample_weight) {
  // sort the data
  // Zip the vectors together
  std::vector<tuple_type> zipped;
  zipped.reserve(len);
  zip<tuple_type>(st, ts, sample_weight, len, zipped);

  // Sort the vector of pairs
  std::sort(std::begin(zipped), std::end(zipped), 
    [&](const auto& a, const auto& b)
    {
        return std::get<1>(a) > std::get<1>(b);
    });

  double fps = 0;
  double tps = 0;
  double last_counted_fps = 0;
  double last_counted_tps = 0;
  double auc = 0;
  for (size_t i=0; i < zipped.size(); ++i) {
    if constexpr(std::is_same<tuple_type, std::tuple<bool, float, float>>::value) {
        tps += std::get<0>(zipped[i]) * std::get<2>(zipped[i]);
        fps += (1 - std::get<0>(zipped[i])) * std::get<2>(zipped[i]);
    }
    else {
        tps += std::get<0>(zipped[i]);
        fps += (1 - std::get<0>(zipped[i]));
    }
    if ((i == zipped.size() - 1) || (std::get<1>(zipped[i+1]) != std::get<1>(zipped[i]))) {
        auc += trapezoid_area(last_counted_fps, fps, last_counted_tps, tps);
        last_counted_fps = fps;
        last_counted_tps = tps;
    }
  }
  return auc / (tps * fps);
}

extern "C" {
    float cpp_auc_ext(float* ts, bool* st, size_t len, float* sample_weight, size_t n_sample_weights) {
        if(n_sample_weights > 0) {
            return auc_kernel<std::tuple<bool, float, float>>(ts, st, len, sample_weight);
        }
        else {
            return auc_kernel<std::tuple<bool, float>>(ts, st, len, sample_weight);
        }
    }
}

template <typename tuple_type>
float aupr_kernel(float* ts, bool* st, size_t len, float* sample_weight) {
    std::vector<tuple_type> zipped;
    zipped.reserve(len);
    zip<tuple_type>(st, ts, sample_weight, len, zipped);

    // Sort by score descending
    std::sort(zipped.begin(), zipped.end(), [](const auto& a, const auto& b) {
        return std::get<1>(a) > std::get<1>(b);
    });

    double tps = 0;
    double fps = 0;
    double last_precision = 0;
    double last_recall = 0;
    double auc_pr = 0;
    double total_positives = 0;

    // Calculate total positives
    for (size_t i = 0; i < len; ++i) {
        if (std::get<0>(zipped[i])) {
            if constexpr (std::is_same<tuple_type, std::tuple<bool, float, float>>::value) {
                total_positives += std::get<2>(zipped[i]);
            } else {
                total_positives += 1;
            }
        }
    }

    for (size_t i = 0; i < len; ++i) {
        if (std::get<0>(zipped[i])) {
            if constexpr (std::is_same<tuple_type, std::tuple<bool, float, float>>::value) {
                tps += std::get<2>(zipped[i]);
            } else {
                tps += 1;
            }
        } else {
            if constexpr (std::is_same<tuple_type, std::tuple<bool, float, float>>::value) {
                fps += std::get<2>(zipped[i]);
            } else {
                fps += 1;
            }
        }

        double precision = tps / (tps + fps);
        double recall = tps / total_positives;

        if (i > 0 && recall != last_recall) {
            auc_pr += (recall - last_recall) * precision;
        }

        last_precision = precision;
        last_recall = recall;
    }

    return auc_pr;
}

extern "C" {
    float cpp_aupr_ext(float* ts, bool* st, size_t len, float* sample_weight, size_t n_sample_weights) {
        if (len == 0) {
            return 0.0f; // No data, return 0 or appropriate value
        }
        if (n_sample_weights > 0) {
            return aupr_kernel<std::tuple<bool, float, float>>(ts, st, len, sample_weight);
        } else {
            return aupr_kernel<std::tuple<bool, float>>(ts, st, len, sample_weight);
        }
    }
}