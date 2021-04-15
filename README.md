# Recursive-Grid-Based-Integration

A grid based integration scheme based on https://asmedigitalcollection.asme.org/computingengineering/article-abstract/18/2/021013/371579/Treat-All-Integrals-as-Volume-Integrals-A-Unified?redirectedFrom=fulltext . This implementation makes use of interval arithmatic along with dynamic parallelism to recursively subdivide the input grid allowing for the exclusion of regions that do not contain non-trivial contributions to the numerical result.

The library PyCu is also needed to evaluate the code and is located here https://github.com/uchytilc/PyCu.
