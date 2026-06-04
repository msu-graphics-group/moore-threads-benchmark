#pragma once
#include "Defs.h"

// Bundles 'IMicrobenchmark' with some test data,
// so we can perform all tests without any knowledge of their nature
class ISealedBenchmark {
  public:
    // Defines how to interpret the measured metrics
    enum class WhoIsBetter {
      LowerIsBetter,  // Lower values are better (e.g., execution time)
      HigherIsBetter, // Higher values are better (e.g., performance in TFlops)
      NeedMinMax      // Both min and max matter for comparison
    };

    ISealedBenchmark(const ISealedBenchmark &) = delete;
    ISealedBenchmark &operator =(const ISealedBenchmark &) = delete;
    virtual ~ISealedBenchmark() = default;

    // The original name of the underlying 'IMicrobenchmark'
    virtual std::string Name() = 0;

    // Resets to the first built-in test configuration
    // After calling this, call 'MoveNext()' to make it active
    virtual void Reset() = 0;

    // Moves to the next test configuration, must be called before the first call of 'Run()'
    // Returns false when no more test configurations are available
    virtual bool MoveNext() = 0;

    // Executes the current test multiple times and returns performance metrics
    // Can throw 'std::exception' if something goes wrong
    // You did not forget to call 'MoveNext()', right?
    virtual std::vector<double> Run() = 0;

    // Returns string representation of the current test configuration
    // Can be used as a caption in tables (e.g., "1024", "24x36", or "pinned-memory")
    virtual std::string Configuration() const = 0;

    // Returns the units of the performance metrics
    // E.g., "TFlops", "GB/s", "seconds", etc
    virtual std::string Units() const = 0;

    // Defines the rule for determining 'better' performance
    // Lower values are better, higher values are better, or both min/max matter
    virtual WhoIsBetter Better() const = 0;
    
  protected:
    ISealedBenchmark() = default;
};
