#pragma once

#include <iostream>

/// Singleton class for thread-safe logging.
class rayMessage {
  std::string message_;

  bool error_ = false;
  constexpr static unsigned tabWidth_ = 4;

  rayMessage() {}

public:
  // delete constructors to result in better error messages by compilers
  rayMessage(const rayMessage &) = delete;
  void operator=(const rayMessage &) = delete;

  static rayMessage &getInstance() {
    static rayMessage instance;
    return instance;
  }

  rayMessage &addWarning(std::string s) {
#pragma omp critical
    {
      message_ += "\n[ViennaRay]" + std::string(tabWidth_, ' ') +
                  "WARNING: " + s + "\n";
    }
    return *this;
  }

  rayMessage &addError(std::string s, bool shouldAbort = true) {
#pragma omp critical
    {
      message_ +=
          "\n[ViennaRay]" + std::string(tabWidth_, ' ') + "ERROR: " + s + "\n";
    }
    // always abort once error message should be printed
    error_ = true;
    // abort now if asked
    if (shouldAbort)
      print();
    return *this;
  }

  rayMessage &addDebug(std::string s) {
#pragma omp critical
    {
      message_ +=
          "[ViennaRay]" + std::string(tabWidth_, ' ') + "DEBUG: " + s + "\n";
    }
    return *this;
  }

  void print(std::ostream &out = std::cout) {
    out << message_;
    message_.clear();
    if (error_)
      abort();
  }
};
