#pragma once
#include "Defs.h"
#include "ISealedBenchmark.h"

namespace helpers {

// Helper to turn a configuration tuple into a human-readable string
template <typename... Args>
std::string TupleToString(const std::tuple<Args...> &t) {
  std::ostringstream oss;
  bool first = true;
  auto printer = [&](const auto &val) {
    if (!first) oss << ", ";
    oss << val;
    first = false;
  };
  std::apply([&](const auto &... args) { (printer(args), ...); }, t);
  return oss.str();
}

} // namespace helpers


// Seals IMicrobenchmark<Args...> into ISealedBenchmark, implemented by DeepSeek-v4
template <typename... Args>
class SealedBenchmark final : public ISealedBenchmark {
  public:
    SealedBenchmark(std::unique_ptr<IMicrobenchmark<Args...>> &&benchmark,
                    const std::vector<std::tuple<Args...>> &configs,
                    size_t iterations,
                    size_t sub_iterations,
                    const std::string &units,
                    WhoIsBetter better)
      : bench_(std::move(benchmark)), configs_(configs),
        iters_(iterations), sub_iters_(sub_iterations),
        units_(units), better_(better)
    { assert(bench_ != nullptr); }

    // The member of 'ISealedBenchmark'
    std::string Name() override { return bench_->Name(); }

    // The member of 'ISealedBenchmark'
    void Reset() override { current_ = -1; }

    // The member of 'ISealedBenchmark'
    bool MoveNext() override {
      ++current_;
      return current_ < static_cast<int>(configs_.size());
    }

    // The member of 'ISealedBenchmark'
    std::vector<double> Run() override {
      assert(0 <= current_ && current_ < (int)configs_.size());
      const auto &args = configs_[current_];
      std::apply([&](const auto &... unpacked) {
        bench_->Configure(iters_, sub_iters_, unpacked...);
      }, args);

      std::vector<double> results;
      bench_->Init();
      try {
        results = bench_->Run();
      }
      catch (const std::exception &) {
        bench_->CleanUp();
        throw;
      }
      bench_->CleanUp();
      return results;
    }

    // The member of 'ISealedBenchmark'
    std::string Configuration() const override {
      assert(0 <= current_ && current_ < (int)configs_.size());
      return helpers::TupleToString(configs_[current_]);
    }

    // The member of 'ISealedBenchmark'
    std::string Units() const override { return units_; }

    // The member of 'ISealedBenchmark'
    WhoIsBetter Better() const override { return better_; }

  private:
    std::unique_ptr<IMicrobenchmark<Args...>> bench_;
    std::vector<std::tuple<Args...>> configs_;
    size_t iters_{ 0 };
    size_t sub_iters_{ 0 };
    int current_{ -1 };
    std::string units_;
    WhoIsBetter better_;
};
