#ifndef RAY_BOUNDCONDITION_HPP
#define RAY_BOUNDCONDITION_HPP

enum struct rayTraceBoundary : unsigned {
  REFLECTIVE = 0,
  PERIODIC = 1,
  IGNORE = 2
};

#endif // RAY_BOUNDCONDITION_HPP