#ifndef RT_RANDOMNUMBERGENERATOR_HPP
#define RT_RANDOMNUMBERGENERATOR_HPP

#include <memory>

// Realization of the i_rng interface is intended to be a random number
// generator together with the definition of a struct which holds all the
// state (e.g. seeds) which the random number generator needs. That is,
// this interface defines how a stateless random number generator relates
// to a state which is held by the user of this interface.
class rtRandomNumberGenerator
{
public:
    virtual ~rtRandomNumberGenerator() {}

    // A definition of the interface of a state
    struct rtRNGState
    {
        virtual ~rtRNGState() {}
        virtual std::unique_ptr<rtRNGState> clone() const = 0;
    };

    // A definition of this function will most likely alter the content of its
    // argument.
    virtual uint64_t get(rtRNGState &pState) const = 0;
    virtual uint64_t min() const = 0;
    virtual uint64_t max() const = 0;
};

#endif // RT_RANDOMNUMBERGENERATOR_HPP