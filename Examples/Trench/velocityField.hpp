#ifndef VELOCITY_FIELD_HPP
#define VELOCITY_FIELD_HPP

#include <lsSmartPointer.hpp>
#include <lsVelocityField.hpp>
#include <unordered_map>
#include <vector>
#include "velocityField.hpp"

template <class T, int D>
class velocityField : public lsVelocityField<T>
{
private:
    typedef std::unordered_map<unsigned long, unsigned long> TranslatorType;

    lsSmartPointer<std::vector<T>> mcestimates = nullptr;
    lsSmartPointer<TranslatorType> translator = nullptr;

public:
    velocityField() {}

    velocityField(lsSmartPointer<std::vector<T>> passedMcEstimates,
                  lsSmartPointer<TranslatorType> passedTranslator)
        : mcestimates(passedMcEstimates), translator(passedTranslator) {}

    T getScalarVelocity(const std::array<T, D> & /*coordinate*/, int material,
                        const std::array<T, D> & /*normalVector*/,
                        unsigned long pointID)
    {

        assert(translator->find(pointID) != translator->end() && "Invalid pointId");

        if (auto it = translator->find(pointID); it != translator->end())
        {
            return mcestimates->operator[](it->second);
        }
        else
        {
            lsMessage::getInstance().addError("velocityField: Invalid pointId");
            return 0.;
        }
    }

    void setTranslator(lsSmartPointer<TranslatorType> passedTranslator)
    {
        translator = passedTranslator;
    }

    void setMcEstimates(lsSmartPointer<std::vector<T>> passedMcEstimates)
    {
        mcestimates = passedMcEstimates;
    }
};

#endif // RT_VELOCITY_FIELD_HPP