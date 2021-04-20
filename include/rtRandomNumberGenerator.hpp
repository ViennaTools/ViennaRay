#ifndef RT_RANDOMNUMBERGENERATOR_HPP
#define RT_RANDOMNUMBERGENERATOR_HPP

#include <memory>
#include <random>

// Realization of the i_rng interface is intended to be a random number
// generator together with the definition of a struct which holds all the
// state (e.g. seeds) which the random number generator needs. That is,
// this interface defines how a stateless random number generator relates
// to a state which is held by the user of this interface.
class rtRandomNumberGenerator
{
public:
    // A definition of the interface of a state
    struct RNGState
    {
        RNGState() : RNGState(std::mt19937_64::default_seed) {}
        RNGState(unsigned int seed) : MT(seed) {}
        RNGState(std::mt19937_64 passedMT) : MT(passedMT) {}

        std::unique_ptr<RNGState> clone()
        {
            return std::make_unique<RNGState>(MT);
        }

        std::mt19937_64 MT;
    };

    // A definition of this function will most likely alter the content of its
    // argument.
    uint64_t get(RNGState &pState)
    {
      return (uint64_t) pState.MT();
    }

    uint64_t min()
    {
        return std::mt19937_64::min();
    }

    uint64_t max()
    {
        return std::mt19937_64::max();
    }
};

#endif // RT_RANDOMNUMBERGENERATOR_HPP