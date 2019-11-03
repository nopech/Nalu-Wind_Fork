/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef HIGHERORDERMASTERELEMENTFUNCTIONS_H
#define HIGHERORDERMASTERELEMENTFUNCTIONS_H

#include <master_element/MasterElement.h>

#include <element_promotion/TensorProductQuadratureRule.h>
#include <element_promotion/LagrangeBasis.h>
#include <element_promotion/NodeMapMaker.h>



#include <AlgTraits.h>
#include <KokkosInterface.h>

#include <array>

namespace sierra {
namespace nalu {

void gradient_2d(
    int nodesPerElement,
    const double* POINTER_RESTRICT elemNodalCoords,
    const double* POINTER_RESTRICT shapeDeriv,
    double* POINTER_RESTRICT grad,
    double* POINTER_RESTRICT det_j);

void gradient_3d(
    int nodesPerElement,
    const double* POINTER_RESTRICT elemNodalCoords,
    const double* POINTER_RESTRICT shapeDeriv,
    double* POINTER_RESTRICT grad,
    double* POINTER_RESTRICT det_j);

double parametric_distance_quad(const double* x);
double parametric_distance_hex(const double* x);

KOKKOS_FUNCTION
int ip_per_face(const TensorProductQuadratureRule& quad, const LagrangeBasis& basis);


} // namespace nalu
} // namespace Sierra

#endif /* HIGHERORDERMASTERELEMENTFUNCTIONS_H */

