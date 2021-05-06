#ifndef RT_MESSAGE_HPP
#define RT_MESSAGE_HPP

#include <iostream>

/// Singleton class for thread-safe logging.
class rtMessage
{
    std::string message;

    bool error = false;
    const unsigned tabWidth = 2;

    rtMessage() {}

public:
    // delete constructors to result in better error messages by compilers
    rtMessage(const rtMessage &) = delete;
    void operator=(const rtMessage &) = delete;

    static rtMessage &getInstance()
    {
        static rtMessage instance;
        return instance;
    }

    rtMessage &addWarning(std::string s)
    {
#pragma omp critical
        {
            message += "\n" + std::string(tabWidth, ' ') + "WARNING: " + s + "\n";
        }
        return *this;
    }

    rtMessage &addError(std::string s, bool shouldAbort = true)
    {
#pragma omp critical
        {
            message += "\n" + std::string(tabWidth, ' ') + "ERROR: " + s + "\n";
        }
        // always abort once error message should be printed
        error = true;
        // abort now if asked
        if (shouldAbort)
            print();
        return *this;
    }

    void print(std::ostream &out = std::cout)
    {
        out << message;
        message.clear();
        if (error)
            abort();
    }
};

#endif // RT_MESSAGE_HPP