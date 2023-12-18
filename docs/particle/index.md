---
layout: default
title: Particles
nav_order: 5
has_children: true
---

# Particles
{: .fs-9 .fw-700}
```c++
#include <rayParticle.hpp>
```
---

## Particle Initialization
Particles are initialized at random positions on the source
plane with random directions according to particle-specific distributions.

Coming Soon
{: .label .label-yellow}

```c++
virtual void initNew(rayRNG &Rng) override {}
```

## Initial Distribution of Directions

Coming Soon
{: .label .label-yellow}

```c++
virtual NumericType getSourceDistributionPower() const { return 1.; }
```

## Data Containers

Coming Soon
{: .label .label-yellow}

```c++
virtual std::vector<std::string> getLocalDataLabels() const {
    return {};
}
virtual void logData(rayDataLog<NumericType> &log) {}
```

## Surface Collision

Coming Soon
{: .label .label-yellow}

| Parameter                | Description                                              | Type                              |
|---------------------------|---------------------------------------------------------|-----------------------------------|
| `rayWeight`               | Current weight of the ray.                              | `NumericType`                     |
| `rayDir`                  | Direction of the ray.                                          | `rayTriple<NumericType>`          |
| `geomNormal`              | Surface normal at the point of collision.                      | `rayTriple<NumericType>`          |
| `primID`                  | Identifier for the primitive being collided with.              | `unsigned int`                    |
| `materialId`              | Identifier for the material of the collided primitive.         | `int`                             |
| `localData`               | Reference to user-defined ray tracing data.                    | `rayTracingData<NumericType>&`    |
| `globalData`              | Pointer to global ray tracing data.                            | `const rayTracingData<NumericType>*` |
| `Rng`                     | Reference to a thread-safe random number generator.            | `rayRNG&`                         |

## Surface Reflection

Coming Soon
{: .label .label-yellow}

| Parameter                | Description                                              | Type                              |
|---------------------------|---------------------------------------------------------|-----------------------------------|
| `rayWeight`               | Current weight of the ray.                              | `NumericType`                     |
| `rayDir`                  | Direction of the ray before reflection.                 | `rayTriple<NumericType>`          |
| `geomNormal`              | Surface normal at the point of reflection.                 | `rayTriple<NumericType>`          |
| `primId`                  | Identifier for the primitive being intersected.            | `unsigned int`                    |
| `materialId`              | Identifier for the material of the intersected primitive.  | `int`                             |
| `globalData`              | Pointer to global ray tracing data.                        | `const rayTracingData<NumericType>*` |
| `Rng`                     | Reference to a thread-safe random number generator.        | `rayRNG&`                         |
