#pragma once
#include "Defs.h"

#include "IMicrobenchmark.h"
#include "ISealedBenchmark.h"
#include "SealedBenchmark.h"


namespace helpers {

template <typename T>
struct is_not_tuple : std::true_type {};

template <typename... Ts>
struct is_not_tuple<std::tuple<Ts...>> : std::false_type {};

template <typename T>
inline constexpr bool is_not_tuple_v = is_not_tuple<T>::value;

} // namespace helpers


// Detects available CUDA devices, manages and runs microbenchmarks
// May be configured for various scenarios
class Framework {
  public:
    Framework();

    // Sets the fraction of iterations reserved for warm-up and outlier handling
    // These iterations are excluded from the final benchmark results
    void ExcludeIterations(double warm_up_fraction, double outlier_fraction);

    // Sets new stream for user-friendly text output
    // By default, this stream is disabled
    void SetTextStream(std::ostream &text) { text_stream_ = std::ref(text); }

    // Sets new stream for CSV output
    // By default, this stream is disabled
    void SetCsvStream(std::ostream &csv) { csv_stream_ = std::ref(csv); }

    // Sets new tag that will be assigned to all following benchmarks
    // Benchmarks with the same tag will be grouped together in the output
    void SetTag(const std::string &tag) { current_tag_ = tag; }

    // Primary template: accepts configurations as vector<tuple<Args...>>
    template <typename... Args>
    void AddBenchmark(std::unique_ptr<IMicrobenchmark<Args...>> &&benchmark,
                      const std::vector<std::tuple<Args...>> &configs,
                      size_t iterations, size_t sub_iterations,
                      Unit units, const std::function<double(double, Args...)> &seconds_to_units,
                      WhoIsBetter better) {
      assert(iterations != 0);
      assert(sub_iterations != 0);
      auto sealed_benchmark = std::unique_ptr<ISealedBenchmark>(new SealedBenchmark<Args...>(
        std::move(benchmark), configs, iterations, sub_iterations, units, seconds_to_units, better
      ));
      benchmarks_[current_tag_].emplace_back(std::move(sealed_benchmark));
    }

    // Convenience overload for benchmarks that take exactly ONE argument
    // Coded by DeepSeek-v4
    template <typename T, typename = std::enable_if_t<helpers::is_not_tuple_v<T>>>
    void AddBenchmark(std::unique_ptr<IMicrobenchmark<T>> &&benchmark,
                      const std::vector<T> &configs,
                      size_t iterations, size_t sub_iterations,
                      Unit units, const std::function<double(double, T)> &seconds_to_units,
                      WhoIsBetter better) {
      std::vector<std::tuple<T>> tuple_configs;
      tuple_configs.reserve(configs.size());
      for (const auto &val : configs) {
        tuple_configs.emplace_back(val);
      }
      AddBenchmark(std::move(benchmark), tuple_configs, iterations,
                   sub_iterations, units, seconds_to_units, better);
    }

    // Convenience overload for benchmarks that require ZERO arguments
    // Coded by DeepSeek-v4
    void AddBenchmark(std::unique_ptr<IMicrobenchmark<>> &&benchmark,
                      size_t iterations, size_t sub_iterations,
                      Unit units, const std::function<double(double)> &seconds_to_units) {
      assert(iterations != 0);
      assert(sub_iterations != 0);
      std::vector<std::tuple<>> configs(1);
      auto sealed_benchmark = std::unique_ptr<ISealedBenchmark>(
        new SealedBenchmark<>(std::move(benchmark), configs,
                              iterations, sub_iterations, units, seconds_to_units, WhoIsBetter::NeedMinMax));
      benchmarks_[current_tag_].emplace_back(std::move(sealed_benchmark));
    }

    // Runs all benchmarks that were added, immediately prints the results to streams
    void Run();

  private:
    double warm_up_fraction_{}, outlier_fraction_{};
    std::reference_wrapper<std::ostream> text_stream_;
    std::reference_wrapper<std::ostream> csv_stream_;

    std::string current_tag_;
    std::unordered_map<std::string, std::vector<std::unique_ptr<ISealedBenchmark>>> benchmarks_;
};
