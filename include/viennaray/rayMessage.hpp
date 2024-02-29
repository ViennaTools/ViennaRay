#ifndef RAY_MESSAGE_HPP
#define RAY_MESSAGE_HPP

#include <iostream>

/// Singleton class for thread-safe logging.
class rayMessage {
  std::string message;

  bool error = false;
  const unsigned tabWidth = 2;

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
      message +=
          "\n[ViennaRay]" + std::string(tabWidth, ' ') + "WARNING: " + s + "\n";
    }
    return *this;
  }

  rayMessage &addError(std::string s, bool shouldAbort = true) {
#pragma omp critical
    {
      message +=
          "\n[ViennaRay]" + std::string(tabWidth, ' ') + "ERROR: " + s + "\n";
    }
    // always abort once error message should be printed
    error = true;
    // abort now if asked
    if (shouldAbort)
      print();
    return *this;
  }

  rayMessage &addDebug(std::string s) {
#pragma omp critical
    {
      message +=
          "[ViennaRay]" + std::string(tabWidth, ' ') + "DEBUG: " + s + "\n";
    }
    return *this;
  }

  void print(std::ostream &out = std::cout) {
    out << message;
    message.clear();
    if (error)
      abort();
  }
};

#endif // RAY_MESSAGE_HPP