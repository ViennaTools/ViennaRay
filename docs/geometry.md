---
layout: default
title: Geometry
nav_order: 4
---
<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']]
  }
};
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
</script>

# Geometry
{: .fs-9 .fw-700 }

---

## Surface

In ViennaRay, the geometry is represented by an oriented point cloud. At each point, a disk is positioned with an orientation that aligns its normal with the surface normal. The disk's radius, denoted as $d_r$, is determined by the grid spacing $d_g$.
The default setting for $d_r$ ensures that the surface is closed which may lead to partially overlapping disks and thus intersections can occur on multiple disks. A depiction of a surface represented by oriented disks is shown below.

<style>
* {
  box-sizing: border-box;
}

.column {
  float: left;
  width: 50%;
  padding: 5px;
}

/* Clearfix (clear floats) */
.row::after {
  content: "";
  clear: both;
  display: table;
}
</style>

<div class="row">
  <div class="column">
    <img src="{% link assets/images/disks.svg %}" alt="2DDisk" style="width:100%">
    Representation of the surface using disks (lines) in 2D. <a href="https://www.iue.tuwien.ac.at/phd/klemenschits/10.html#autosec-148">(Image Source)</a>
  </div>
  <div class="column">
    <img src="" alt="3DDisks" style="width:100%">
    Representation of a 3D surface using oriented disks.
  </div>
</div>

The geometry in ViennaRay can be set through the `rayTrace` class, using the member function `setGeometry`, which is explained in detail [here]({% link tracer.md %}).

## Source Plane

Coming Soon
{: .label .label-yellow}

## Boundary Conditions

Coming Soon
{: .label .label-yellow}