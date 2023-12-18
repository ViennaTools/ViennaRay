---
layout: default
title: Tracer
nav_order: 6
---

# Tracer
{: .fs-9 .fw-700}
```c++
#include <rayTrace.hpp>
```
---

Coming Soon
{: .label .label-yellow}

## Setting the Geometry

```c++
void setGeometry(std::vector<std::array<NumericType, Dim>> &points,
                std::vector<std::array<NumericType, Dim>> &normals,
                const NumericType gridDelta)
```

```c++
template <typename T> void setMaterialIds(std::vector<T> &pMaterialIds) 
```

```c++
void setBoundaryConditions(rayBoundaryCondition pBoundaryConditions[D])
```

```c++
void setSourceDirection(const rayTraceDirection pDirection)
```

<details markdown="1">

<summary> 
<b>Example usage</b> 
</summary>

```c++
...
rayTrace<NumericType, 3> tracer;
tracer.setGeometry(points, normals, gridDelta);
tracer.setMaterialIds(matIds);
tracer.setSourceDirection(rayTraceDirection::POS_Z);

rayBoundaryCondition boundardyConds[3] = {rayBoundaryCondition::REFLECTIVE};
tracer.setBoundaryConditions(boundaryConds);
...
```

</details>

## Setting the Particle
```c++
template <typename ParticleType>
void setParticleType(std::unique_ptr<ParticleType> &p)
```
Set the particle type used for ray tracing. The particle is a user defined object that has to interface the `rayParticle` class.

```c++
void setNumberOfRaysPerPoint(const size_t pNum)
void setNumberOfRaysFixed(const size_t pNum)
```
Set the number of rays per geometry point. The total number of traced rays is determined by multiplying the set number with the total number of points in the geometry. Alternatively, you can fix the total number of traced rays, regardless of the geometry.

```c++
void setPrimaryDirection(const rayTriple<NumericType> pPrimaryDirection)
```
Set the primary direction of the source distribution. This can be used to obtain a tilted source distribution. Setting the primary direction does not change the position of the source plane. Therefore, one has to be careful that the resulting distribution does not lie completely above the source plane.

<!-- Particle, num rays per point, primary directions -->
<details markdown="1">

<summary> 
<b>Example usage</b> 
</summary>

```c++
...
rayTrace<NumericType, 3> tracer;

auto myParticle = std::make_unique<myParticleType>();
tracer.setParticleType(myParticle);
tracer.setNumRaysPerPoint(1000);
...
```
</details>


## Local Data and Global Data

## Extracting the Results

## Additional Settings